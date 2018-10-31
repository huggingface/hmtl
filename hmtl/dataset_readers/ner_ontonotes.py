# coding: utf-8

import logging
from typing import Dict, List, Iterable

from overrides import overrides

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


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ner_ontonotes")
class NerOntonotesReader(DatasetReader):
    '''
    An ``allennlp.data.dataset_readers.dataset_reader.DatasetReader`` for reading
    NER annotations in CoNll-formatted OntoNotes dataset.
    
    NB: This DatasetReader was implemented before the current implementation of 
    ``OntonotesNamedEntityRecognition`` in AllenNLP. It is thought doing pretty much the same thing.
    
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Map a token to an id.
    domain_identifier : ``str``, optional (default = None)
        The subdomain to load. If None is specified, the whole dataset is loaded.
    label_namespace : ``str``, optional (default = "ontonotes_ner_labels")
        The tag/label namespace for the task/dataset considered.
    lazy : ``bool``, optional (default = False)
        Whether or not the dataset should be loaded in lazy way. 
        Refer to https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/laziness.md
        for more details about lazyness.
    coding_scheme: ``str``, optional (default=``IOB1``)
        Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
        Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
        the original IOB1 scheme in the CoNLL data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    '''
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 label_namespace: str = "ontonotes_ner_labels",
                 lazy: bool = False,
                 coding_scheme: str = "IOB1") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier
        self._label_namespace = label_namespace
        self._coding_scheme = coding_scheme
        if coding_scheme not in ("IOB1", "BIOUL"):
            raise ConfigurationError("unknown coding_scheme: {}".format(coding_scheme))
        
    @overrides
    def _read(self,
              file_path: str):
        file_path = cached_path(file_path) # if `file_path` is a URL, redirect to the cache
        ontonotes_reader = Ontonotes()
        logger.info("Reading NER instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)
            
        for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.named_entities:
                tags = ["O" for _ in tokens]
            else:
                tags = sentence.named_entities
                
            if self._coding_scheme == "BIOUL":
                tags = iob1_to_bioul(tags)
                
            yield self.text_to_instance(tokens, tags)
          
        
    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str,
                          domain_identifier: str) -> Iterable[OntonotesSentence]:
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            yield from ontonotes_reader.sentence_iterator(conll_file)
    
    
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
                