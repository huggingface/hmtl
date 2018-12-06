from overrides import overrides

from allennlp.training.metrics import ConllCorefScores


class ConllCorefFullScores(ConllCorefScores):
    """
    This is marginal modification of the class ``allennlp.training.metrics.metric.ConllCorefScores``.
    It leaves the possibility to get the 3 detailled coreference metrics (B3, MUC, CEAFE),
    and not only their average.
    """

    def __init__(self) -> None:
        super(ConllCorefFullScores, self).__init__()

    @overrides
    def get_metric(self, reset: bool = False, full: bool = False):
        full_metrics = {}
        if full:
            for e in self.scorers:
                metric_name = e.metric.__name__
                full_metrics[metric_name] = {
                    "precision": e.get_precision(),
                    "recall": e.get_recall(),
                    "f1_score": e.get_f1(),
                }

        metrics = (lambda e: e.get_precision(), lambda e: e.get_recall(), lambda e: e.get_f1())
        precision, recall, f1_score = tuple(
            sum(metric(e) for e in self.scorers) / len(self.scorers) for metric in metrics
        )

        full_metrics["coref_precision"] = precision
        full_metrics["coref_recall"] = recall
        full_metrics["coref_f1"] = f1_score

        if reset:
            self.reset()

        return full_metrics
