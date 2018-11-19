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

from hmtl.models.relation_extraction import RelationExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("relation")
class LayerRelation(Model):
    """
    A class that implement one task of HMTL model: Relation Extraction.
    
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
                
        super(LayerRelation, self).__init__(vocab = vocab, regularizer = regularizer)

        # Base text Field Embedder
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, 
                                                                params=text_field_embedder_params)
        self._text_field_embedder = text_field_embedder
        
        ############################
        # Relation Extraction Stuffs
        ############################
        relation_params = params.pop("relation")
        
        # Encoder
        encoder_relation_params = relation_params.pop("encoder")
        encoder_relation = Seq2SeqEncoder.from_params(encoder_relation_params)
        self._encoder_relation =  encoder_relation
        
        # Tagger: Relation
        tagger_relation_params = relation_params.pop("tagger")
        tagger_relation = RelationExtractor(vocab = vocab,
                                            text_field_embedder = self._text_field_embedder,
                                            context_layer = self._encoder_relation,
                                            d = tagger_relation_params.pop_int("d"),
                                            l = tagger_relation_params.pop_int("l"),
                                            n_classes = tagger_relation_params.pop("n_classes"),
                                            activation = tagger_relation_params.pop("activation"))
        self._tagger_relation = tagger_relation	

        logger.info("Multi-Task Learning Model has been instantiated.")

    @overrides		
    def forward(self, 
                tensor_batch,
                for_training: bool = False,
                task_name: str = "relation") -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        
        tagger = getattr(self, "_tagger_%s" % task_name)
        return tagger.forward(**tensor_batch)

    @overrides		
    def get_metrics(self,
                    task_name: str = "relation",
                    reset: bool = False,
                    full: bool = False) -> Dict[str, float]:
        
        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset)

    @classmethod    
    def from_params(cls,
                    vocab: Vocabulary,
                    params: Params,
                    regularizer: RegularizerApplicator) -> "LayerRelation":
        return cls(vocab = vocab,
                params = params,
                regularizer = regularizer)
        