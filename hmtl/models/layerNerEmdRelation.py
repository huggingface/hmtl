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

from hmtl.modules.text_field_embedders import ShortcutConnectTextFieldEmbedder
from hmtl.models.relation_extraction import RelationExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("ner_emd_relation")
class LayerNerEmdRelation(Model):
    """
    A class that implement three tasks of HMTL model: NER (CRF Tagger), EMD (CRF Tagger) and Relation Extraction.
    
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
                
        super(LayerNerEmdRelation, self).__init__(vocab = vocab, regularizer = regularizer)
        
        # Base text Field Embedder
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
        
        
        ############
        # EMD Stuffs
        ############
        emd_params = params.pop("emd")
        
        # Encoder
        encoder_emd_params = emd_params.pop("encoder")
        encoder_emd = Seq2SeqEncoder.from_params(encoder_emd_params)
        self._encoder_emd =  encoder_emd
        
        shortcut_text_field_embedder = ShortcutConnectTextFieldEmbedder(base_text_field_embedder = self._text_field_embedder,
                                                                        previous_encoders = [self._encoder_ner])
        self._shortcut_text_field_embedder = shortcut_text_field_embedder
        
        
        # Tagger: EMD - CRF Tagger
        tagger_emd_params = emd_params.pop("tagger")
        tagger_emd = CrfTagger(vocab = vocab,
                                text_field_embedder = self._shortcut_text_field_embedder,
                                encoder = self._encoder_emd,
                                label_namespace = tagger_emd_params.pop("label_namespace", "labels"),
                                constraint_type = tagger_emd_params.pop("constraint_type", None),
                                dropout = tagger_ner_params.pop("dropout", None),
                                regularizer = regularizer)
        self._tagger_emd = tagger_emd
        
        
        ############################
        # Relation Extraction Stuffs
        ############################
        relation_params = params.pop("relation")
        
        # Encoder
        encoder_relation_params = relation_params.pop("encoder")
        encoder_relation = Seq2SeqEncoder.from_params(encoder_relation_params)
        self._encoder_relation =  encoder_relation
        
        shortcut_text_field_embedder_relation = ShortcutConnectTextFieldEmbedder(base_text_field_embedder = self._text_field_embedder,
                                                                                previous_encoders = [self._encoder_ner, self._encoder_emd])
        self._shortcut_text_field_embedder_relation = shortcut_text_field_embedder_relation
        
        # Tagger: Relation
        tagger_relation_params = relation_params.pop("tagger")
        tagger_relation = RelationExtractor(vocab = vocab,
                                            text_field_embedder = self._shortcut_text_field_embedder_relation,
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
                task_name: str = "ner") -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        
        tagger = getattr(self, "_tagger_%s" % task_name)
        return tagger.forward(**tensor_batch)
            
    @overrides
    def get_metrics(self,
                    task_name: str,
                    reset: bool = False,
                    full: bool = False) -> Dict[str, float]:
        
        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset)

    @classmethod    
    def from_params(cls,
                    vocab: Vocabulary,
                    params: Params,
                    regularizer: RegularizerApplicator) -> "LayerNerEmdRelation":
        return cls(vocab = vocab,
                params = params,
                regularizer = regularizer)