# coding: utf-8

import logging
from typing import Dict, List, Iterable, Iterator

from overrides import overrides
import codecs

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import iob1_to_bioul
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence

from hmtl.dataset_readers.dataset_utils import ACE, ACESentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


        
@DatasetReader.register("mention_ace")
class MentionACEReader(DatasetReader):
    '''
    A dataset reader to read the Entity Mention Tags from an ACE dataset
    previously pre-procesed to fit the CoNll-NER format.
    '''
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_namespace: str = "ace_mention_labels",
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_namespace = label_namespace
    
    
    @staticmethod
    def _sentence_iterate(ace_reader: ACE,
                        file_path: str) -> Iterable[ACESentence]:
        for conll_file in ace_reader.dataset_path_iterator(file_path):
            yield from ace_reader.sentence_iterator(conll_file)
    
    
    @overrides
    def _read(self,
              file_path: str):
        file_path = cached_path(file_path) # if `file_path` is a URL, redirect to the cache
        ace_reader = ACE()
        logger.info("Reading ACE Mention instances from dataset files at: %s", file_path)
        
        for sentence in self._sentence_iterate(ace_reader, file_path):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.mention_tags:
                tags = ["O" for _ in tokens]
            else:
                tags = sentence.mention_tags

            yield self.text_to_instance(tokens, tags)
    
    
    def text_to_instance(self,
                         tokens: List[Token],
                         tags: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        if tags:
            fields['tags'] = SequenceLabelField(labels = tags, sequence_field = text_field, label_namespace = self._label_namespace)
        return Instance(fields)
         