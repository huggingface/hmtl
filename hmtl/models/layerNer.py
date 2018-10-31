# coding: utf-8

import os
import sys
import logging
from typing import Dict
from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import RegularizerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models.crf_tagger import CrfTagger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

        
@Model.register("ner")
class LayerNer(Model):
    """
    A class that implement the first task of HMTL model: NER (CRF Tagger).
    
    Parameters
    ----------
    vocab: ``allennlp.data.Vocabulary``, required.
        The vocabulary fitted on the data.
    params: ``allennlp.common.Params``, required
        Configuration parameters for the multi-task model.
    regularizer: ``allennlp.nn.RegularizerApplicator``, optional (default = None)
        A reguralizer to apply to the model's layers.
    """
    def __init__(self,
                vocab: Vocabulary,
                params: Params,
                regularizer: RegularizerApplicator = None):
                
        super(LayerNer, self).__init__(vocab = vocab, regularizer = regularizer)

        # Base Text Field Embedder
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, 
                                                                params=text_field_embedder_params)
        self._text_field_embedder = text_field_embedder
        
        ############
        # NER Stuffs
        ############
        ner_params = params.pop("ner")
        
        # Encoder
        encoder_ner_params = ner_params.pop("encoder")
        encoder_ner = Seq2SeqEncoder.from_params(encoder_ner_params)
        self._encoder_ner =  encoder_ner
        
        # Tagger NER - CRF Tagger
        tagger_ner_params = ner_params.pop("tagger")
        tagger_ner = CrfTagger(vocab = vocab,
                            text_field_embedder = self._text_field_embedder,
                            encoder = self._encoder_ner,
                            label_namespace = tagger_ner_params.pop("label_namespace", "labels"),
                            constraint_type = tagger_ner_params.pop("constraint_type", None),
                            dropout = tagger_ner_params.pop("dropout", None),
                            regularizer = regularizer)
        self._tagger_ner = tagger_ner
        
        logger.info("Multi-Task Learning Model has been instantiated.")
    
    @overrides    
    def forward(self, 
                tensor_batch,
                for_training: bool = False,
                task_name: str = "ner") -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        
        tagger = getattr(self, "_tagger_%s" % task_name)
        return tagger.forward(**tensor_batch)
        
    @overrides
    def get_metrics(self,
                    task_name: str = "ner",
                    reset: bool = False,
                    full: bool = False) -> Dict[str, float]:
        
        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset = reset)
    
    @classmethod    
    def from_params(cls,
                    vocab: Vocabulary,
                    params: Params,
                    regularizer: RegularizerApplicator) -> "LayerNer":
        return cls(vocab = vocab,
                params = params,
				regularizer = regularizer)
        