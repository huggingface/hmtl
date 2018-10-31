# coding: utf-8

"""
A quick and simple script for evaluating the embeddings throught the HTML model/hierarchy
using SentEval.
"""


from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import re

# Set PATHs
PATH_TO_SENTEVAL = './SentEval/'
PATH_TO_DATA = './SentEval/data'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

import os
import torch
import argparse

from allennlp.common.params import Params
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import Token, Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.nn import util
from allennlp.models.model import Model

import hmtl


def text_to_instance(sent, token_indexers):
    text = TextField([Token(word) for word in sent], token_indexers = token_indexers)
    instance = Instance({"text": text})
    return instance

def sentences_to_indexed_batch(sentences, token_indexers):
    instances = [text_to_instance(sent, token_indexers) for sent in sentences]
    batch = Batch(instances)
    batch.index_instances(vocab)
    return batch 	
    
def compute_embds_from_layer(model, model_layer_name, batch):
    batch_tensor = batch.as_tensor_dict(batch.get_padding_lengths())
    text = batch_tensor["text"]
    text_mask = util.get_text_field_mask(text)
    
    if model_layer_name == "text_field_embedder":
        embds_text_field_embedder = model._text_field_embedder(text)
        embds = embds_text_field_embedder
        
    if model_layer_name == "encoder_ner":
        embds_text_field_embedder = model._text_field_embedder(text)
        embds_encoder_ner = model._encoder_ner(embds_text_field_embedder, text_mask)
        embds = embds_encoder_ner
        
    if model_layer_name == "encoder_emd":
        embds_text_field_embedder = model._shortcut_text_field_embedder(text)
        embds_encoder_emd = model._encoder_emd(embds_text_field_embedder, text_mask)
        embds = embds_encoder_emd
        
    if model_layer_name == "encoder_relation":
        embds_text_field_embedder = model._shortcut_text_field_embedder_relation(text)
        embds_encoder_relation = model._encoder_relation(embds_text_field_embedder, text_mask)
        embds = embds_encoder_relation
    
    if model_layer_name == "encoder_coref":
        embds_text_field_embedder = model._shortcut_text_field_embedder_coref(text)
        embds_encoder_coref = model._encoder_coref(embds_text_field_embedder, text_mask)
        embds = embds_encoder_coref
    
    emds_size = embds.size(2)
    expanded_text_mask = torch.cat([text_mask.unsqueeze(-1)]*emds_size, dim = -1)
        
    embds_sum = (embds*expanded_text_mask.float()).sum(dim = 1)
    normalization = torch.cat([(1/text_mask.float().sum(-1)).unsqueeze(-1)]*emds_size, dim = -1)
    computed_embds = (embds_sum*normalization)
    
    return computed_embds.detach().numpy()


# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = sentences_to_indexed_batch(batch, token_index)
    embds = compute_embds_from_layer(model, args.layer_name, batch)
    return embds


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}


# Set up logger
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":	
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",
                        "--serialization_dir",
                        required = True,
                        help = "Directory from which to load the pretrained model.", 
                        type = str)
    parser.add_argument("-t",
                        "--task",
                        required = False,
                        default = "ner",
                        help = "Name of the task to load.", 
                        type = str)	
    parser.add_argument("-l",
                        "--layer_name",
                        required = False,
                        default = "text_field_embedder",
                        help = "Name of encoder/embedding layer of the model", 
                        type = str)										
    args = parser.parse_args()
    
    
    serialization_dir = args.serialization_dir

    params = Params.from_file(params_file = os.path.join(args.serialization_dir, "config.json"))
    logging.info("Parameters loaded from %s", os.path.join(serialization_dir, "config.json"))
    
    ### Load Vocabulary from files ###
    logging.info("Loading Vocavulary from %s", os.path.join(serialization_dir, "vocabulary"))
    vocab = Vocabulary.from_files(os.path.join(args.serialization_dir, "vocabulary"))
    logger.info("Vocabulary loaded")
    
    ### Create model ###
    model_params = params.pop("model")
    model = Model.from_params(vocab = vocab, params = model_params, regularizer = None)
    best_model_state_path = os.path.join(serialization_dir, "best_{}.th".format(args.task))
    best_model_state = torch.load(best_model_state_path)
    model.load_state_dict(state_dict = best_model_state)
    
    ### Create token indexer ###
    token_index = {}
    task_keys = [key for key in params.keys() if re.search("^task_", key)] 
    token_indexer_params = params.pop(task_keys[-1]).pop("data_params").pop("dataset_reader").pop("token_indexers")
    for name, indexer_params in token_indexer_params.items(): 
        token_index[name] = TokenIndexer.from_params(indexer_params) 
    
    params_senteval['encoder'] = model
    
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    
    print(results)
    logging.info("SentEval(uation) Finished")
