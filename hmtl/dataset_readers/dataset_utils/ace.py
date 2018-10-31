# coding: utf-8

from typing import DefaultDict, List, Optional, Iterator, Set, Tuple
from collections import defaultdict
import codecs
import os
import logging

from allennlp.data.dataset_readers.dataset_utils import iob1_to_bioul

from nltk import Tree

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TypedSpan = Tuple[int, Tuple[int, int]]  # pylint: disable=invalid-name
TypedStringSpan = Tuple[str, Tuple[int, int]]  # pylint: disable=invalid-name

class ACESentence:
    """
    A class representing the annotations available for a single ACE CONLL-formatted sentence.

    Parameters
    ----------
    words : ``List[str]``
        This is the tokens as segmented/tokenized with spayc.
    mention_tags : ``List[str]``
        The BIO tags for Entity Mention Detection in the sentence.
    relations : ``List[Tuple[str, List[str]]]``
        The relations tags for Relation Extraction in the sentence.
    last_head_token_relations : ``List[Tuple[str, List[str]]]``
        The relations tags between last tokens for ARG1 and ARG2 for Relation Extraction in the sentence.
    coref_spans : ``Set[TypedSpan]``
        The spans for entity mentions involved in coreference resolution within the sentence.
        Each element is a tuple composed of (cluster_id, (start_index, end_index)). Indices
        are `inclusive`.
    """
    def __init__(self,
                words: List[str],
                mention_tags: List[str],
                relations: List[Tuple[str, List[str]]],
                last_head_token_relations: List[Tuple[str, List[str]]],
                coref_spans: Set[TypedSpan]):
        self.words = words
        self.mention_tags = mention_tags
        self.relations = relations
        self.last_head_token_relations = last_head_token_relations
        self.coref_spans = coref_spans


class ACE:
    """
    This DatasetReader is designed to read in the ACE (2005 or 2004) which
    have been previously formatted in the format used by the CoNLL format
    (see for instance OntoNotes dataset).
    """
    def dataset_iterator(self, file_path: str) -> Iterator[ACESentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        """
        for conll_file in self.dataset_path_iterator(file_path):
            yield from self.sentence_iterator(conll_file)

    @staticmethod
    def dataset_path_iterator(file_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory
        containing CONLL-formatted files.
        """
        logger.info("Reading ACE CONLL-like sentences from dataset files at: %s", file_path)
        for root, _, files in list(os.walk(file_path)):
            for data_file in files:
                if not data_file.endswith("like_conll"):
                    continue

                yield os.path.join(root, data_file)

    def dataset_document_iterator(self, file_path: str) -> Iterator[List[ACESentence]]:
        """
        An iterator over CONLL-formatted files which yields documents, regardless
        of the number of document annotations in a particular file.
        """
        with codecs.open(file_path, 'r', encoding='utf8') as open_file:
            conll_rows = []
            document: List[ACESentence] = []
            for line in open_file:
                line = line.strip()
                if line != '' and not line.startswith('#'):
                    # Non-empty line. Collect the annotation.
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        document.append(self._conll_rows_to_sentence(conll_rows))
                        conll_rows = []
                if line.startswith("#end document"):
                    yield document
                    document = []
            if document:
                # Collect any stragglers or files which might not
                # have the '#end document' format for the end of the file.
                yield document

    def sentence_iterator(self, file_path: str) -> Iterator[ACESentence]:
        """
        An iterator over the sentences in an individual CONLL formatted file.
        """
        for document in self.dataset_document_iterator(file_path):
            for sentence in document:
                yield sentence

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> ACESentence:
        sentence: List[str] = []
        mention_tags: List[str] = []
        
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []
        
        # Cluster id -> List of (start_index, end_index) spans.
        clusters: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        # Cluster id -> List of start_indices which are open for this id.
        coref_stacks: DefaultDict[int, List[int]] = defaultdict(list)
        
        for index, row in enumerate(conll_rows):
            conll_components = row.split()
            
            word = conll_components[1]
            
            if not span_labels:
                span_labels = [[] for _ in conll_components[2:-1]]
                current_span_labels = [None for _ in conll_components[2:-1]]
            self._process_span_annotations_for_word(annotations = conll_components[2:-1],
                                                    span_labels = span_labels,
                                                    current_span_labels = current_span_labels)
            
            #Process coref
            self._process_coref_span_annotations_for_word(conll_components[-1],
                                                index,
                                                clusters,
                                                coref_stacks)

            sentence.append(word)
        
            
        mention_tags = iob1_to_bioul(span_labels[0])
        
        #Process coref clusters
        coref_span_tuples: Set[TypedSpan] = {(cluster_id, span)
                                for cluster_id, span_list in clusters.items()
                                for span in span_list}
        
        
        #Reformat the labels to only keep the the last token of the head
        #Cf paper, we model relation between last tokens of heads.
        last_head_token_relations = []
        bioul_relations = []

        for relation_frame in span_labels[1:]:
            bioul_relation_frame = iob1_to_bioul(relation_frame)
            
            reformatted_frame = []
            for annotation in bioul_relation_frame:
                if annotation[:2] in ["L-", "U-"]: 
                    reformatted_frame.append(annotation[2:])
                else: 
                    reformatted_frame.append("*")
                    
            last_head_token_relations.append(reformatted_frame)
            bioul_relations.append(bioul_relation_frame)

        return ACESentence(sentence, mention_tags, bioul_relations, last_head_token_relations, coref_span_tuples)
        
        
    @staticmethod
    def _process_mention_tags(annotations: List[str]):
        """
        Read and pre-process the entity mention tags as a formatted in CoNll-NER-style.
        """
        labels = []
        current_span_label = None
        for annotation in annotations:
            label = annotation.strip("()*")
            if "(" in annotation:
                bio_label = "B-" + label
                current_span_label = label
            elif current_span_label is not None:
                bio_label = "I-" + current_span_label
            else:
                bio_label = "O"
            if ")" in annotation:
                current_span_label = None
            labels.append(bio_label)
        return labels
        
    @staticmethod
    def _process_span_annotations_for_word(annotations: List[str],
                                           span_labels: List[List[str]],
                                           current_span_labels: List[Optional[str]]) -> None:
        """
        Given a sequence of different label types for a single word and the current
        span label we are inside, compute the BIO tag for each label and append to a list.

        Parameters
        ----------
        annotations: ``List[str]``
            A list of labels to compute BIO tags for.
        span_labels : ``List[List[str]]``
            A list of lists, one for each annotation, to incrementally collect
            the BIO tags for a sequence.
        current_span_labels : ``List[Optional[str]]``
            The currently open span per annotation type, or ``None`` if there is no open span.
        """
        for annotation_index, annotation in enumerate(annotations):
            # strip all bracketing information to
            # get the actual propbank label.
            label = annotation.strip("()*")

            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                bio_label = "B-" + label
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                # If there's no '(' token, but the current_span_label is not None,
                # then we are inside a span.
                bio_label = "I-" + current_span_labels[annotation_index]
                span_labels[annotation_index].append(bio_label)
            else:
                # We're outside a span.
                span_labels[annotation_index].append("O")
            # Exiting a span, so we reset the current span label for this annotation.
            if ")" in annotation:
                current_span_labels[annotation_index] = None
                
                
    @staticmethod
    def _process_coref_span_annotations_for_word(label: str,
                                                 word_index: int,
                                                 clusters: DefaultDict[int, List[Tuple[int, int]]],
                                                 coref_stacks: DefaultDict[int, List[int]]) -> None:
        """
        For a given coref label, add it to a currently open span(s), complete a span(s) or
        ignore it, if it is outside of all spans. This method mutates the clusters and coref_stacks
        dictionaries.

        Parameters
        ----------
        label : ``str``
            The coref label for this word.
        word_index : ``int``
            The word index into the sentence.
        clusters : ``DefaultDict[int, List[Tuple[int, int]]]``
            A dictionary mapping cluster ids to lists of inclusive spans into the
            sentence.
        coref_stacks: ``DefaultDict[int, List[int]]``
            Stacks for each cluster id to hold the start indices of active spans (spans
            which we are inside of when processing a given word). Spans with the same id
            can be nested, which is why we collect these opening spans on a stack, e.g:

            [Greg, the baker who referred to [himself]_ID1 as 'the bread man']_ID1
        """
        if label != "-":
            for segment in label.split("|"):
                # The conll representation of coref spans allows spans to
                # overlap. If spans end or begin at the same word, they are
                # separated by a "|".
                if segment[0] == "(":
                    # The span begins at this word.
                    if segment[-1] == ")":
                        # The span begins and ends at this word (single word span).
                        cluster_id = int(segment[1:-1])
                        clusters[cluster_id].append((word_index, word_index))
                    else:
                        # The span is starting, so we record the index of the word.
                        cluster_id = int(segment[1:])
                        coref_stacks[cluster_id].append(word_index)
                else:
                    # The span for this id is ending, but didn't start at this word.
                    # Retrieve the start index from the document state and
                    # add the span to the clusters for this id.
                    cluster_id = int(segment[:-1])
                    start = coref_stacks[cluster_id].pop()
                    clusters[cluster_id].append((start, word_index))