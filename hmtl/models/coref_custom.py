import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import MentionRecall, ConllCorefScores
from allennlp.models.coreference_resolution import CoreferenceResolver

from hmtl.training.metrics import ConllCorefFullScores

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CoreferenceCustom(CoreferenceResolver):
    """
    This class implements a marginally modified version of ``allennlp.models.coreference_resolution.CoreferenceResolver``
    which is an implementation of the model of Lee et al., 2017.
    The two modifications are:
        1/ Replacing the scorer to be able to get the 3 detailled coreference metrics (B3, MUC, CEAFE),
        and not only their average.
        2/ Give the possibility to evaluate with the gold mentions: the model first predict mentions that MIGHT
        be part of a coreference cluster, and in second time predict the coreference clusters for theses mentions.
        We leave the possibility of replacing predicting the possible mentions 
        with the gold mentions in evaluation.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        mention_feedforward: FeedForward,
        antecedent_feedforward: FeedForward,
        feature_size: int,
        max_span_width: int,
        spans_per_word: float,
        max_antecedents: int,
        lexical_dropout: float = 0.2,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        eval_on_gold_mentions: bool = False,
    ) -> None:
        super(CoreferenceCustom, self).__init__(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            context_layer=context_layer,
            mention_feedforward=mention_feedforward,
            antecedent_feedforward=antecedent_feedforward,
            feature_size=feature_size,
            max_span_width=max_span_width,
            spans_per_word=spans_per_word,
            max_antecedents=max_antecedents,
            lexical_dropout=lexical_dropout,
            initializer=initializer,
            regularizer=regularizer,
        )

        self._conll_coref_scores = ConllCorefFullScores()
        self._eval_on_gold_mentions = eval_on_gold_mentions

        if self._eval_on_gold_mentions:
            self._use_gold_mentions = False
        else:
            self._use_gold_mentions = None

    @overrides
    def get_metrics(self, reset: bool = False, full: bool = False):
        mention_recall = self._mention_recall.get_metric(reset=reset)
        metrics = self._conll_coref_scores.get_metric(reset=reset, full=full)
        metrics["mention_recall"] = mention_recall

        return metrics

    @overrides
    def forward(
        self,  # type: ignore
        text: Dict[str, torch.LongTensor],
        spans: torch.IntTensor,
        span_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        document_length = text_embeddings.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text).float()

        # Shape: (batch_size, num_spans)
        if self._use_gold_mentions:
            if text_embeddings.is_cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            s = [
                torch.as_tensor(pair, dtype=torch.long, device=device)
                for cluster in metadata[0]["clusters"]
                for pair in cluster
            ]
            gm = torch.stack(s, dim=0).unsqueeze(0).unsqueeze(1)

            span_mask = spans.unsqueeze(2) - gm
            span_mask = (span_mask[:, :, :, 0] == 0) + (span_mask[:, :, :, 1] == 0)
            span_mask, _ = (span_mask == 2).max(-1)
            num_spans = span_mask.sum().item()
            span_mask = span_mask.float()
        else:
            span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
            num_spans = spans.size(1)
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, document_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, emebedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Prune based on mention scores.
        num_spans_to_keep = int(math.floor(self._spans_per_word * document_length))

        (top_span_embeddings, top_span_mask, top_span_indices, top_span_mention_scores) = self._mention_pruner(
            span_embeddings, span_mask, num_spans_to_keep
        )
        top_span_mask = top_span_mask.unsqueeze(-1)
        # Shape: (batch_size * num_spans_to_keep)
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans, top_span_indices, flat_top_span_indices)

        # Compute indices for antecedent spans to consider.
        max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = self._generate_valid_antecedents(
            num_spans_to_keep, max_antecedents, util.get_device_of(text_mask)
        )
        # Select tensors relating to the antecedent spans.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        candidate_antecedent_embeddings = util.flattened_index_select(top_span_embeddings, valid_antecedent_indices)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        candidate_antecedent_mention_scores = util.flattened_index_select(
            top_span_mention_scores, valid_antecedent_indices
        ).squeeze(-1)
        # Compute antecedent scores.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = self._compute_span_pair_embeddings(
            top_span_embeddings, candidate_antecedent_embeddings, valid_antecedent_offsets
        )
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = self._compute_coreference_scores(
            span_pair_embeddings,
            top_span_mention_scores,
            candidate_antecedent_mention_scores,
            valid_antecedent_log_mask,
        )

        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_antecedents = coreference_scores.max(2)
        predicted_antecedents -= 1

        output_dict = {
            "top_spans": top_spans,
            "antecedent_indices": valid_antecedent_indices,
            "predicted_antecedents": predicted_antecedents,
        }
        if span_labels is not None:
            # Find the gold labels for the spans which we kept.
            pruned_gold_labels = util.batched_index_select(
                span_labels.unsqueeze(-1), top_span_indices, flat_top_span_indices
            )

            antecedent_labels = util.flattened_index_select(pruned_gold_labels, valid_antecedent_indices).squeeze(-1)
            antecedent_labels += valid_antecedent_log_mask.long()

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            gold_antecedent_labels = self._compute_antecedent_gold_labels(pruned_gold_labels, antecedent_labels)
            coreference_log_probs = util.masked_log_softmax(coreference_scores, top_span_mask)
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).sum()

            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(top_spans, valid_antecedent_indices, predicted_antecedents, metadata)

            output_dict["loss"] = negative_marginal_log_likelihood

        if metadata is not None:
            output_dict["document"] = [x["original_text"] for x in metadata]
        return output_dict
