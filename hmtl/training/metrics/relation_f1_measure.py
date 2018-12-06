from typing import Dict, List, Optional, Set
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask  # , ones_like
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric


@Metric.register("relation_f1")
class RelationF1Measure(Metric):
    """
    """

    def __init__(self) -> None:
        """
        A class for computing the metrics specific to relation extraction.
        We consider a relation correct if we correctly predict the last of the head of the two arguments and the relation type.
        """
        self._true_positives: int = 0
        self._false_positives: int = 0
        self._false_negatives: int = 0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Update the TP, FP and FN counters.
        
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)  # ones_like(gold_labels)
        # Get the data from the Variables.
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        if gold_labels.size() != predictions.size():
            raise ConfigurationError("Predictions and gold labels don't have the same size.")

        # Apply mask
        # Compute the mask before computing the loss
        # Transform the mask that is at the sentence level (#Size: n_batches x padded_document_length)
        # to a suitable format for the relation labels level
        _, padded_document_length, _, n_classes = predictions.size()
        mask = mask.float()
        squared_mask = torch.stack([e.view(padded_document_length, 1) * e for e in mask], dim=0)
        squared_mask = squared_mask.unsqueeze(-1).repeat(
            1, 1, 1, n_classes
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes

        gold_labels = gold_labels.cpu()

        predictions = (
            predictions * squared_mask
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        gold_labels = (
            gold_labels * squared_mask
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            flattened_predictions = predictions[i].view(-1).nonzero().cpu().numpy()
            flattened_gold_labels = gold_labels[i].view(-1).nonzero().cpu().numpy()

            for prediction in flattened_predictions:
                if prediction in flattened_gold_labels:
                    self._true_positives += 1
                else:
                    self._false_positives += 1
            for gold in flattened_gold_labels:
                if gold not in flattened_predictions:
                    self._false_negatives += 1

    def get_metric(self, reset: bool = False):
        """
        Get the metrics and reset the counters if necessary.
        """
        all_metrics = {}

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            self._true_positives, self._false_positives, self._false_negatives
        )
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0
