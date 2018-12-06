# coding: utf-8

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable  # from torch.nn.parameter import Parameter, Variable

from overrides import overrides

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util

from hmtl.training.metrics import RelationF1Measure

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# Mapping specific to the dataset used in our setting (ACE2005)
# Please adapt it if necessary
rel_type_2_idx = {"ORG-AFF": 0, "PHYS": 1, "ART": 2, "PER-SOC": 3, "PART-WHOLE": 4, "GEN-AFF": 5}
idx_2_rel_type = {value: key for key, value in rel_type_2_idx.items()}


@Model.register("relation_extractor")
class RelationExtractor(Model):
    """
	A class containing the scoring model for relation extraction.
	It is derived the model proposed by Bekoulis G. in 
	"Joint entity recognition and relation extraction as a multi-head selection problem"
	https://bekou.github.io/papers/eswa2018b/bekoulis_eswa_2018b.pdf
	
	Parameters
	----------
	vocab: ``allennlp.data.Vocabulary``, required.
        The vocabulary fitted on the data.
	text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``, required
        This layer incorporates contextual information for each word in the document.
	d: ``int``, required
		The (half) dimension of embedding given	by the encoder context_layer.
	l: ``int``, required
		The dimension of the relation extractor scorer embedding.
	n_classes: ``int``, required
		The number of different possible relation classes.
	activation: ``str``, optional (default = "relu")
		Non-linear activation function for the scorer. Can be either "relu" or "tanh".
	label_namespace: ``str``, optional (default = "relation_ace_labels")
		The namespace for the labels of the task of relation extraction.
	"""

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        d: int,
        l: int,
        n_classes: int,
        activation: str = "relu",
        label_namespace: str = "relation_ace_labels",
    ) -> None:
        super(RelationExtractor, self).__init__(vocab)

        self._U = nn.Parameter(torch.Tensor(2 * d, l))
        self._W = nn.Parameter(torch.Tensor(2 * d, l))
        self._V = nn.Parameter(torch.Tensor(l, n_classes))
        self._b = nn.Parameter(torch.Tensor(l))

        self.init_weights()

        self._n_classes = n_classes
        self._activation = activation

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

        self._label_namespace = label_namespace

        self._relation_metric = RelationF1Measure()

        self._loss_fn = nn.BCEWithLogitsLoss()

    def init_weights(self) -> None:
        """
		Initialization for the weights of the model.
		"""
        nn.init.kaiming_normal_(self._U)
        nn.init.kaiming_normal_(self._W)
        nn.init.kaiming_normal_(self._V)

        nn.init.normal_(self._b)

    def multi_class_cross_entropy_loss(self, scores, labels, mask):
        """
		Compute the loss from
		"""
        # Compute the mask before computing the loss
        # Transform the mask that is at the sentence level (#Size: n_batches x padded_document_length)
        # to a suitable format for the relation labels level
        padded_document_length = mask.size(1)
        mask = mask.float()  # Size: n_batches x padded_document_length
        squared_mask = torch.stack([e.view(padded_document_length, 1) * e for e in mask], dim=0)
        squared_mask = squared_mask.unsqueeze(-1).repeat(
            1, 1, 1, self._n_classes
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes

        # The scores (and gold labels) are flattened before using
        # the binary cross entropy loss.
        # We thus transform
        flat_size = scores.size()
        scores = scores * squared_mask  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        scores_flat = scores.view(
            flat_size[0], flat_size[1], flat_size[2] * self._n_classes
        )  # Size: n_batches x padded_document_length x (padded_document_length x n_classes)
        labels = labels * squared_mask  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        labels_flat = labels.view(
            flat_size[0], flat_size[1], flat_size[2] * self._n_classes
        )  # Size: n_batches x padded_document_length x (padded_document_length x n_classes)

        loss = self._loss_fn(scores_flat, labels_flat)

        # Amplify the loss to actually see something...
        return 100 * loss

    @overrides
    def forward(self, text: Dict[str, torch.LongTensor], relations: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
		Forward pass of the model.
		Compute the predictions and the loss (if labels are available).
		
		Parameters:
		----------
		text: Dict[str, torch.LongTensor]
			The input sentences which have transformed into indexes (integers) according to a mapping token:str -> token:int
		relations: torch.IntTensor
			The gold relations to predict.
		"""

        # Text field embedder map the token:int to their word embedding representation token:embedding (whatever these embeddings are).
        text_embeddings = self._text_field_embedder(text)
        # Compute the mask from the text: 1 if there is actually a word in the corresponding sentence, 0 if it has been padded.
        mask = util.get_text_field_mask(text)  # Size: batch_size x padded_document_length

        # Compute the contextualized representation from the word embeddings.
        # Usually, _context_layer is a Seq2seq model such as LSTM
        encoded_text = self._context_layer(
            text_embeddings, mask
        )  # Size: batch_size x padded_document_length x lstm_output_size

        ###### Relation Scorer ##############
        # Compute the relation scores
        left = torch.matmul(encoded_text, self._U)  # Size: batch_size x padded_document_length x l
        right = torch.matmul(encoded_text, self._W)  # Size: batch_size x padded_document_length x l

        left = left.permute(1, 0, 2)
        left = left.unsqueeze(3)
        right = right.permute(0, 2, 1)
        right = right.unsqueeze(0)

        B = left + right
        B = B.permute(1, 0, 3, 2)  # Size: batch_size x padded_document_length x padded_document_length x l

        outer_sum_bias = B + self._b  # Size: batch_size x padded_document_length x padded_document_length x l
        if self._activation == "relu":
            activated_outer_sum_bias = F.relu(outer_sum_bias)
        elif self._activation == "tanh":
            activated_outer_sum_bias = F.tanh(outer_sum_bias)

        relation_scores = torch.matmul(
            activated_outer_sum_bias, self._V
        )  # Size: batch_size x padded_document_length x padded_document_length x n_classes
        #################################################################

        batch_size, padded_document_length = mask.size()

        relation_sigmoid_scores = torch.sigmoid(
            relation_scores
        )  # F.sigmoid(relation_scores) #Size: batch_size x padded_document_length x padded_document_length x n_classes

        # predicted_relations[l, i, j, k] == 1 iif we predict a relation k with ARG1==i, ARG2==j in the l-th sentence of the batch
        predicted_relations = torch.round(
            relation_sigmoid_scores
        )  # Size: batch_size x padded_document_length x padded_document_length x n_classes

        output_dict = {
            "relation_sigmoid_scores": relation_sigmoid_scores,
            "predicted_relations": predicted_relations,
            "mask": mask,
        }

        if relations is not None:
            # Reformat the gold relations before computing the loss
            # Size: batch_size x padded_document_length x padded_document_length x n_classes
            # gold_relations[l, i, j, k] == 1 iif we predict a relation k with ARG1==i, ARG2==j in the l-th sentence of the batch
            gold_relations = torch.zeros(batch_size, padded_document_length, padded_document_length, self._n_classes)

            for exple_idx, exple_tags in enumerate(relations):  # going through the batch
                # rel is a list of list containing the current sentence in the batch
                # each sublist in rel is of size padded_document_length
                # and encodes a relation in the sentence where the two non zeros elements
                # indicate the two words arguments AND the relation type between these two words.
                for rel in exple_tags:
                    # relations have been padded, so for each sentence in the batch there are
                    # max_nb_of_relations_in_batch_for_one_sentence relations ie (number of sublist such as rel)
                    # The padded relations are simply list of size padded_document_length filled with 0.
                    if rel.sum().item() == 0:
                        continue

                    for idx in rel.nonzero():
                        label_srt = self.vocab.get_token_from_index(rel[idx].item(), self._label_namespace)
                        arg, rel_type = label_srt.split("_")
                        if arg == "ARG1":
                            x = idx.data[0]
                        else:
                            y = idx.data[0]

                    gold_relations[exple_idx, x, y, rel_type_2_idx[rel_type]] = 1

                    # GPU support
            if text_embeddings.is_cuda:
                gold_relations = gold_relations.cuda()

            # Compute the loss
            output_dict["loss"] = self.multi_class_cross_entropy_loss(
                scores=relation_scores, labels=gold_relations, mask=mask
            )

            # Compute the metrics with the predictions.
            self._relation_metric(predictions=predicted_relations, gold_labels=gold_relations, mask=mask)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
		Decode the predictions
		"""
        decoded_predictions = []

        for instance_tags in output_dict["predicted_relations"]:
            sentence_length = instance_tags.size(0)
            decoded_relations = []

            for arg1, arg2, rel_type_idx in instance_tags.nonzero().data:
                relation = ["*"] * sentence_length
                rel_type = idx_2_rel_type[rel_type_idx.item()]
                relation[arg1] = "ARG1_" + rel_type
                relation[arg2] = "ARG2_" + rel_type
                decoded_relations.append(relation)

            decoded_predictions.append(decoded_relations)

        output_dict["decoded_predictions"] = decoded_predictions

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
		Compute the metrics for relation: precision, recall and f1.
		A relation is considered correct if we can correctly predict the last word of ARG1, the last word of ARG2 and the relation type.
		"""
        metric_dict = self._relation_metric.get_metric(reset=reset)
        return {x: y for x, y in metric_dict.items() if "overall" in x}
