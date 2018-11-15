# coding: utf-8

import os
import argparse
from typing import List, Dict, Any, Iterable
import torch
import torch.nn.functional as F
import math
import spacy
import re
from emoji import UNICODE_EMOJI

from allennlp.models.model import Model
from allennlp.common.params import Params
from allennlp.data import Vocabulary, Token, Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField, Field, ListField, SpanField
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.nn import util

import sys
sys.path.append('../')
import hmtl
from predictionFormatter import predictionFormatter

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


torch.set_num_threads(1)

MAX_STRING_SIZE = 500
COREF_MAX_SPAN_WIDTH = 8

   

def is_only_emoji(text):
    '''
    Test if an incoming text is only composed of emojis.
    '''
    result = True
    for c in text:
        if c not in UNICODE_EMOJI: 
            result = False
            break
    return result
    
    
def filter_messages(input_text, sent): 
    '''
    Filter messages which are either too long or only emojis.
    '''   
    # Filter messages which are not long enough or too long
    if len(input_text) >= MAX_STRING_SIZE:
        return True
        
    # Filter messages which are only emojis
    if is_only_emoji(input_text):
        return True
            
    return False


def create_instance(sentence, vocab, token_indexers):
    '''
    Create an batch tensor from the input sentence.
    '''
    text = TextField([Token(word) for word in sentence], token_indexers = token_indexers)
    
    spans = []
    for start, end in enumerate_spans(sentence,
                                    offset=0,
                                    max_span_width=COREF_MAX_SPAN_WIDTH):
        spans.append(SpanField(start, end, text))
    span_field = ListField(spans)
    
    instance = Instance({"tokens": text, "spans": span_field})
    
    instances =  [instance]
    batch = Batch(instances)
    batch.index_instances(vocab)
    batch_tensor = batch.as_tensor_dict(batch.get_padding_lengths())
    
    return batch_tensor
    
    
def load_model(model_name = "conll_full_elmo"):
    '''
    Load both vocabulary and model and create and instance of
    HMTL full model.
    '''
    if model_name not in ["conll_small_elmo", "conll_medium_elmo", "conll_full_elmo"]:
        raise ValueError(f"{model_name} is not a valid name of model.")
    serialization_dir = "model_dumps" + "/" + model_name
    params = Params.from_file(params_file = os.path.join(serialization_dir, "config.json"))
    
    # Load TokenIndexer
    task_keys = [key for key in params.keys() if re.search("^task_", key)]
    token_indexer_params = params.pop(task_keys[-1]).pop("data_params").pop("dataset_reader").pop("token_indexers")
    #see https://github.com/allenai/allennlp/issues/181 for better syntax
    token_indexers = {} 
    for name, indexer_params in token_indexer_params.items(): 
        token_indexers[name] = TokenIndexer.from_params(indexer_params) 
    
    # Load the vocabulary
    logger.info("Loading Vocavulary from %s", os.path.join(serialization_dir, "vocabulary"))
    vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
    logger.info("Vocabulary loaded")
    
    # Create model and load weights
    model_params = params.pop("model")
    model = Model.from_params(vocab = vocab, params = model_params, regularizer = None)
    model_state_path = os.path.join(serialization_dir, "weights.th")
    model_state = torch.load(model_state_path, map_location = "cpu")
    model.load_state_dict(state_dict = model_state)
    
    return model, vocab, token_indexers
    
    
    
    
class HMTLPredictor:
    '''
    Predictor class for HMTL full model.
    '''
    def __init__(self, model_name = "conll_full_elmo"):
        model, vocab, token_indexers = load_model(model_name = model_name)
        self.model = model
        self.vocab = vocab
        self.token_indexers = token_indexers
        self.formatter = predictionFormatter()
        self.nlp = spacy.load("en_core_web_sm")
    
    def predict(self, input_text, raw_format = False):  
        '''
        Take an input text and compute its prediction with HMTL model.
        If sentence is 2 tokens or less, coreference is not called.
        '''      
        with torch.no_grad():
            self.model.eval()
            
            ### Prepare batch ###
            input_text = input_text.strip()
            sent, sent_char_offset, doc = self.parse_text(input_text = input_text)
                   
            message_filtered = filter_messages(input_text = input_text, sent = sent)
                 
            if message_filtered:
                final_output = self.fallback_prediction(input_text = input_text, sent = sent)
            else:
                if len(sent) < 3: required_tasks = ["ner", "emd", "relation"]
                else: required_tasks = ["ner", "emd", "relation", "coref"]

                batch_tensor = create_instance(sentence = sent, vocab = self.vocab, token_indexers = self.token_indexers)            
                final_output = self.inference(batch = batch_tensor, required_tasks = required_tasks)
                final_output["tokenized_text"] = sent
                
                if "coref" not in final_output.keys(): final_output["coref"] = [[]]
            
            if not raw_format:
                final_output = self.formatter.format(final_output, sent_char_offset, input_text)
                final_output = self.formatter.expand(final_output, doc)
                
            return message_filtered, final_output
        
    # def inference(self,
    #             tensor_batch,
    #             task_name: str = "emd"):
    #     # pylint: disable=arguments-differ
        
    #     tagger = getattr(self, "_tagger_%s" % task_name)
    #     output = tagger.forward(**tensor_batch)
        
    #     decoding_dict = tagger.decode(output)
    #     return decoding_dict
        
        
    def inference(self, batch, required_tasks):
        '''
        Fast inference of HMTL.
        '''
        # pylint: disable=arguments-differ
        
        final_output = {}
        
        ### Fast inference of NER ###
        output_ner, embedded_text_input_base, encoded_text_ner, mask = self.inference_ner(batch)
        decoding_dict_ner = self.decode(task_output = output_ner, task_name = "ner")
        final_output["ner"] = decoding_dict_ner["tags"]
        
        ### Fast inference of EMD ###
        output_emd, _, encoded_text_emd, mask = self.inference_emd(embedded_text_input_base, encoded_text_ner, mask)
        decoding_dict_emd = self.decode(task_output = output_emd, task_name = "emd")
        final_output["emd"] = decoding_dict_emd["tags"]
        
        ### Fast inference of Relation ###
        output_relation, embedded_text_input_relation, mask = self.inference_relation(embedded_text_input_base, encoded_text_emd, mask)  
        decoding_dict_relation = self.decode(task_output = output_relation, task_name = "relation")
        final_output["relation"] = decoding_dict_relation["decoded_predictions"]
        
        ### Fast inference of Coreference ##
        if "coref" in required_tasks:
            output_coref = self.inference_coref(batch, embedded_text_input_relation, mask)  
            decoding_dict_coref = self.decode(task_output = output_coref, task_name = "coref")
            final_output["coref"] = decoding_dict_coref["clusters"]
        
        return final_output
        
        
    def inference_ner(self, batch):
        submodel = self.model._tagger_ner
        
        ### Fast inference of NER ###
        tokens = batch["tokens"]
        embedded_text_input_base = submodel.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        
        encoded_text_ner = submodel.encoder(embedded_text_input_base, mask)
        
        logits = submodel.tag_projection_layer(encoded_text_ner)
        best_paths = submodel.crf.viterbi_tags(logits, mask)
        
        predicted_tags = [x for x, y in best_paths]
        
        output = {"tags": predicted_tags}
        
        return output, embedded_text_input_base, encoded_text_ner, mask
        
        
    def inference_emd(self, embedded_text_input_base, encoded_text_ner, mask):
        submodel = self.model._tagger_emd
        
        ### Fast inference of EMD ###
        embedded_text_input_emd = torch.cat([embedded_text_input_base, encoded_text_ner], dim = -1)
        
        encoded_text_emd = submodel.encoder(embedded_text_input_emd, mask)
        
        logits = submodel.tag_projection_layer(encoded_text_emd)
        best_paths = submodel.crf.viterbi_tags(logits, mask)
        
        predicted_tags = [x for x, y in best_paths]
        
        output = {"tags": predicted_tags}
        
        return output, embedded_text_input_emd, encoded_text_emd, mask
        
        
    def inference_relation(self, embedded_text_input_base, encoded_text_emd, mask):
        submodel = self.model._tagger_relation
        
        ### Fast inference of Relation ###
        embedded_text_input_relation = torch.cat([embedded_text_input_base, encoded_text_emd], dim = -1)
        
        encoded_text_relation = submodel._context_layer(embedded_text_input_relation, mask)
        
        left = torch.matmul(encoded_text_relation, submodel._U)
        right = torch.matmul(encoded_text_relation, submodel._W)
        left = left.permute(1,0,2)
        left = left.unsqueeze(3)
        right = right.permute(0,2,1)
        right = right.unsqueeze(0)
        B = left + right
        B = B.permute(1,0,3,2)
        
        outer_sum_bias = B + submodel._b
        if submodel._activation == "relu":
            activated_outer_sum_bias = F.relu(outer_sum_bias)
        elif submodel._activation == "tanh":
            activated_outer_sum_bias = F.tanh(outer_sum_bias)
            
        relation_scores = torch.matmul(activated_outer_sum_bias, submodel._V)
        relation_sigmoid_scores = torch.sigmoid(relation_scores)
        predicted_relations = torch.round(relation_sigmoid_scores)
        
        output = {"predicted_relations": predicted_relations,
                "relation_sigmoid_scores": relation_sigmoid_scores}
        
        return output, embedded_text_input_relation, mask   
        

    def inference_coref(self, batch, embedded_text_input_relation, mask):
        submodel = self.model._tagger_coref
        
        ### Fast inference of coreference ###
        spans = batch["spans"]
                
        document_length = mask.size(1)
        num_spans = spans.size(1)

        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        spans = F.relu(spans.float()).long()

        encoded_text_coref = submodel._context_layer(embedded_text_input_relation, mask)
        endpoint_span_embeddings = submodel._endpoint_span_extractor(encoded_text_coref, spans)
        attended_span_embeddings = submodel._attentive_span_extractor(embedded_text_input_relation, spans)

        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
        num_spans_to_keep = int(math.floor(submodel._spans_per_word * document_length))

        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores) = submodel._mention_pruner(span_embeddings,
                                                                                        span_mask,
                                                                                        num_spans_to_keep)
        top_span_mask = top_span_mask.unsqueeze(-1)
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)
        top_spans = util.batched_index_select(spans,
                                              top_span_indices,
                                              flat_top_span_indices)

        max_antecedents = min(submodel._max_antecedents, num_spans_to_keep)

        valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = \
            submodel._generate_valid_antecedents(num_spans_to_keep, max_antecedents, util.get_device_of(mask))
        candidate_antecedent_embeddings = util.flattened_index_select(top_span_embeddings,
                                                                      valid_antecedent_indices)

        candidate_antecedent_mention_scores = util.flattened_index_select(top_span_mention_scores,
                                                                          valid_antecedent_indices).squeeze(-1)
        span_pair_embeddings = submodel._compute_span_pair_embeddings(top_span_embeddings,
                                                                  candidate_antecedent_embeddings,
                                                                  valid_antecedent_offsets)
        coreference_scores = submodel._compute_coreference_scores(span_pair_embeddings,
                                                              top_span_mention_scores,
                                                              candidate_antecedent_mention_scores,
                                                              valid_antecedent_log_mask)

        _, predicted_antecedents = coreference_scores.max(2)
        predicted_antecedents -= 1

        output_dict = {"top_spans": top_spans,
                       "antecedent_indices": valid_antecedent_indices,
                       "predicted_antecedents": predicted_antecedents}
                       
        return output_dict  
       
        
    def decode(self, task_output, task_name: str = "ner"):
        '''
        Decode the predictions.
        '''
        tagger = getattr(self.model, "_tagger_%s" % task_name)
        return tagger.decode(task_output)
        
        
    def parse_text(self, input_text):
        '''
        Tokenized the input sentence, extract the tokens and their first character index in the sentence.
        '''
        doc = self.nlp(input_text)
        sent = [word.string.strip() for word in doc]
        sent_char_offset = [word.idx for word in doc]
        return sent, sent_char_offset, doc
    
    
    def fallback_prediction(self, input_text, sent):
        '''
        If message is filtered (message is too long or emoji), return a default API output.
        '''
        return {"text": input_text,
                "tokenized_text": sent,
                "ner": [[]],
                "emd": [[]],
                "relation": [[]],
                "relation_debug": [[]],
                "coref": [[]]}    
    
    
        
if __name__ == "__main__":		
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--model_name",
                        default = "conll_full_elmo",
                        required = False,
                        type = str, 
                        help = 'Name of the model to use.')				
    args = parser.parse_args()
                            
    
    hmtl = HMTLPredictor(model_name = args.model_name)
    
    input_text = "Her sister used to swim with Barack Obama. He is not bad, but she is better. I like watching her."
    output = hmtl.predict(input_text = input_text)
    print(output)    