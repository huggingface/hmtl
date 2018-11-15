# coding: utf-8

from allennlp.data.dataset_readers.dataset_utils.span_utils import bioul_tags_to_spans
import copy

def find_indices(lst, condition):
    '''
    Find the indices of elements in a list `lst` that match a condition.
    '''
    return [i for i, elem in enumerate(lst) if condition(elem)]

def expand_arg(relation, arg_nb, n_c, formatted_relation):
    '''
    Update formatted_relation to take into account the possible expansion of relation arguments.
    '''
    arg_str = "arg{}".format(arg_nb)
    arg = (relation[arg_str + "_begin_char"], 
            relation[arg_str + "_end_char"], 
            relation[arg_str + "_text"])
    if arg in n_c.keys():
        formatted_relation[arg_str + "_begin_char"] = n_c[arg].start_char
        formatted_relation[arg_str + "_end_char"] = n_c[arg].end_char

        formatted_relation[arg_str + "_begin_token"] = n_c[arg].start
        formatted_relation[arg_str + "_end_token"] = n_c[arg].end - 1
        
        formatted_relation[arg_str + "_text"] = n_c[arg].text
    else:
        formatted_relation[arg_str + "_begin_token"] = relation.get(arg_str +  "_index")
        formatted_relation[arg_str + "_end_token"] = relation.get(arg_str + "_index")

def check_overlapping(relation, formatted_relation):
    '''
    Check if there is no overlapping between the two expanded arguments of a relation.
    If there is an overlap, we drop the expansion for the relation.
    '''
    arg1_b, arg1_e = formatted_relation["arg1_begin_char"], formatted_relation["arg1_end_char"]
    arg2_b, arg2_e = formatted_relation["arg2_begin_char"], formatted_relation["arg2_end_char"]
    
    overlap = False \
            or (arg1_b < arg2_e and arg1_e >= arg2_b) \
            or (arg2_b < arg1_e and arg2_e >= arg1_b)
    
    if overlap: 
        arg1_i, arg2_i = relation["arg1_index"], relation["arg2_index"]
        relation["arg1_begin_token"], relation["arg1_end_token"]  = arg1_i, arg1_i
        relation["arg2_begin_token"], relation["arg2_end_token"]  = arg2_i, arg2_i
        return relation
    else:
        return formatted_relation  

class predictionFormatter:
    '''
    A class that format the prediction returned by HMTL model.
    If necessary, it also expands
    '''
    def __init__(self):
        pass
        
    def format(self, predictions, sent_char_offset, input_text):        
        tokenized_text = predictions["tokenized_text"]
        predicted_tasks = predictions.keys()
        
        formatted_predictions = {}
        formatted_predictions["tokenized_text"] = tokenized_text
        
        
        ### Format NER and EMD ###
        for task_name in ["ner", "emd"]:
            if task_name in predicted_tasks:
                decoded_bioul = []
                assert len(predictions[task_name]) == 1
                
                spans = bioul_tags_to_spans(predictions[task_name][0])
                for tag, (begin, end) in spans:
                    entity = {"type": tag, 
                            "begin_token": begin, "end_token": end, 
                            "begin_char": sent_char_offset[begin], "end_char": sent_char_offset[end] + len(tokenized_text[end]),
                            "tokenized_text": tokenized_text[begin:(end+1)],
                            "text": input_text[sent_char_offset[begin]:(sent_char_offset[end] + len(tokenized_text[end]))]}
                    decoded_bioul.append(entity)
                    
                formatted_predictions[task_name] = decoded_bioul
        
        
        ### Format Relation ###
        if "relation" in predicted_tasks:
            decoded_relation_arcs = []
            assert len(predictions["relation"]) == 1
            
            for i, relation in enumerate(predictions["relation"][0]):
                indices = find_indices(relation, lambda x: x!= "*")
                for ind in indices:
                    tag = relation[ind]
                    if tag[:4] == "ARG1": arg1_index, arg1_text = ind, tokenized_text[ind]
                    if tag[:4] == "ARG2": arg2_index, arg2_text = ind, tokenized_text[ind]
                rel = {"type": tag[5:],
                        "arg1_index": arg1_index, "arg1_text": arg1_text, 
                        "arg1_begin_char": sent_char_offset[arg1_index], "arg1_end_char": sent_char_offset[arg1_index] + len(arg1_text),
                        "arg2_index": arg2_index, "arg2_text": arg2_text, 
                        "arg2_begin_char": sent_char_offset[arg2_index], "arg2_end_char": sent_char_offset[arg2_index] + len(arg2_text)}
                decoded_relation_arcs.append(rel)
                
            formatted_predictions["relation_arcs"] = decoded_relation_arcs
        
        
        ### Format Coreference ###
        if "coref" in predicted_tasks:
            decoded_coref_arcs = []
            decoded_coref_clusters = []
            assert len(predictions["coref"]) == 1
            
            for cluster in predictions["coref"][0]:
                ## Format the clusters
                decoded_cluster = []
                for mention in cluster:
                    begin, end = mention
                    m = {"begin": begin, "end": end, 
                        "begin_char": sent_char_offset[begin], "end_char": sent_char_offset[end] + len(tokenized_text[end]),
                        "tokenized_text": tokenized_text[begin:(end+1)],
                        "text": input_text[sent_char_offset[begin]:(sent_char_offset[end] + len(tokenized_text[end]))]}
                    decoded_cluster.append(m)
                decoded_coref_clusters.append(decoded_cluster)
                
                ## Format the arcs
                for i in range(len(cluster)-1):
                    mention1_begin, mention1_end = cluster[i]
                    mention2_begin, mention2_end = cluster[i+1]
                    coref_arc = {"mention1_begin": mention1_begin, "mention1_end": mention1_end, 
                                "mention1_begin_char": sent_char_offset[mention1_begin], "mention1_end_char": sent_char_offset[mention1_end] + len(tokenized_text[mention1_end]),
                                "tokenized_text1": tokenized_text[mention1_begin:(mention1_end+1)],
                                "text1": input_text[sent_char_offset[mention1_begin]:(sent_char_offset[mention1_end] + len(tokenized_text[mention1_end]))],
                                "mention2_begin": mention2_begin, "mention2_end": mention2_end, 
                                "mention2_begin_char": sent_char_offset[mention2_begin], "mention2_end_char": sent_char_offset[mention2_end] + len(tokenized_text[mention2_end]),
                                "tokenized_text2": tokenized_text[mention2_begin:(mention2_end+1)],
                                "text2": input_text[sent_char_offset[mention2_begin]:(sent_char_offset[mention2_end] + len(tokenized_text[mention2_end]))]}
                    decoded_coref_arcs.append(coref_arc)
            
            formatted_predictions["coref_arcs"] = decoded_coref_arcs
            formatted_predictions["coref_clusters"] = decoded_coref_clusters
        
        return formatted_predictions
        
    def expand_relations(self, predictions, doc):
        '''
        HMTL predicts the relation between the last head tokens.
        This is a simple heuristic to expand relations using a dependecy tree.
        '''
        if "relation_arcs" in predictions:            
            predictions["relation_arcs_expanded"] = []
            noun_chunks = {}
            for chunk in doc.noun_chunks: 
                noun_chunks[(chunk.root.idx, chunk.root.idx + len(chunk.root.text), chunk.root.text)] = chunk                      
                    
            for relation in predictions["relation_arcs"]:
                formatted_relation = copy.deepcopy(relation)   
                
                expand_arg(relation, 1, noun_chunks, formatted_relation)
                expand_arg(relation, 2, noun_chunks, formatted_relation)
                formatted_relation = check_overlapping(relation, formatted_relation)
                
                del formatted_relation["arg1_index"], formatted_relation["arg2_index"]
                
                predictions["relation_arcs_expanded"].append(formatted_relation)
                             
        return predictions
        
    def expand_emd(self, predictions, doc):
        '''
        HMTL predicts the heads of a mention.
        Simple heuristic to expand entity mentions using a dependecy tree.
        '''
        if "emd" in predictions:            
            noun_chunks = {}
            for chunk in doc.noun_chunks:
                noun_chunks[(chunk.root.idx, chunk.root.idx + len(chunk.root.text), chunk.root.text)] = chunk
            
            predictions["emd_expanded"] = []
            for emd in predictions["emd"]:
                expanded_emd = copy.deepcopy(emd)
                id_ = (emd["begin_char"], emd["end_char"], emd["text"])
                
                if id_ in noun_chunks.keys():
                    expanded_emd["begin_char"] = noun_chunks[id_].start_char
                    expanded_emd["end_char"] = noun_chunks[id_].end_char
                    
                    expanded_emd["begin_token"] = noun_chunks[id_].start
                    expanded_emd["end_token"] = noun_chunks[id_].end - 1
        
                    expanded_emd["text"] = noun_chunks[id_].text
                    expanded_emd["tokenized_text"] = [token.text for token in noun_chunks[id_]]
                    
                predictions["emd_expanded"].append(expanded_emd)           
        
        return predictions
        
    def expand(self, predictions, doc):
        '''
        Perform both EMD and Relation expansion
        '''
        predictions = self.expand_relations(predictions, doc)
        predictions = self.expand_emd(predictions, doc)
        
        return predictions