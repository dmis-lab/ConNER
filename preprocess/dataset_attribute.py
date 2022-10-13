# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import copy
import math
import json
import glob
import torch
import random
import logging
import argparse
import numpy as np

from tqdm import tqdm, trange

def token_frequency(token_freq_dict, token_cons_dict, words, labels, entity_name):
    temp_token_cons_dict = {}
    for word in words:
        if word not in token_freq_dict:
            token_freq_dict[word] = 0
        token_freq_dict[word] += 1

    for word, label in zip(words, labels):
        if word not in temp_token_cons_dict:
            temp_token_cons_dict[word] = {0:0, 1:0, 2:0, 3:0, 4:0}
        temp_token_cons_dict[word][label] += 1

    for key, val in temp_token_cons_dict.items():
        if key not in token_cons_dict:
            token_cons_dict[key] = []

        total_cnt = 0
        temp_list = []
        for label_id, label_cnt in val.items():
            total_cnt += label_cnt

        if 'bc5cdr' == entity_name:
            temp_cnt, temp_cnt2 = 0, 0
            for label_id, label_cnt in val.items():
                if label_id == 0:
                    temp_list.append(label_cnt / total_cnt)
                elif label_id == 1:
                    temp_cnt += label_cnt
                elif label_id == 2:
                    temp_cnt2 += label_cnt
                elif label_id == 3:
                    temp_cnt += label_cnt
                    temp_list.append(temp_cnt / total_cnt)
                elif label_id == 4:
                    temp_cnt2 += label_cnt
                    temp_list.append(temp_cnt2 / total_cnt)
        else:
            temp_cnt, temp_cnt2 = 0, 0
            for label_id, label_cnt in val.items():
                if label_id == 0:
                    temp_list.append(label_cnt / total_cnt)
                elif label_id == 1:
                    temp_cnt += label_cnt
                elif label_id == 2:
                    temp_cnt += label_cnt
                    temp_list.append(temp_cnt / total_cnt)
            
        token_cons_dict[key].append(temp_list)

    return token_freq_dict, token_cons_dict

def entity_frequency(entity_freq_dict, entity_density_list, entity_cons_dict, words, labels, entity_name):
    entity = ""
    in_entity = ""
    in_entity_freq_dict = {}
    def cnt_function(entity_freq_dict, entity):
        entity = entity.strip()
        if entity not in entity_freq_dict:
            entity_freq_dict[entity] = 0
        entity_freq_dict[entity] += 1
        entity = ""
        return entity_freq_dict, entity

    for idx, (word, label) in enumerate(zip(words, labels)):
        if label != 0:
            if label in [1, 2]:
                entity += word + " "
                in_entity += word + " "
                try:
                    if 'bc5cdr' == entity_name:
                        if labels[idx+1] == 0 or labels[idx+1] == 1 or labels[idx+1] == 2:
                            entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                            in_entity_freq_dict, in_entity = cnt_function(in_entity_freq_dict, in_entity)
                        else:
                            continue
                    else:
                        if labels[idx+1] == 0 or labels[idx+1] == 1:
                            entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                            in_entity_freq_dict, in_entity = cnt_function(in_entity_freq_dict, in_entity)
                        else:
                            continue
                except:
                    entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                    in_entity_freq_dict, in_entity = cnt_function(in_entity_freq_dict, in_entity)
            else:
                entity += word + " "
                in_entity += word + " "
                try:
                    if labels[idx+1] != 0:
                        continue
                    else:                
                        entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                        in_entity_freq_dict, in_entity = cnt_function(in_entity_freq_dict, in_entity)
                except:
                    entity_freq_dict, entity = cnt_function(entity_freq_dict, entity)
                    in_entity_freq_dict, in_entity = cnt_function(in_entity_freq_dict, in_entity)
                
    # get a density of entity per document
    doc_len = len(words)
    temp_entity_length_dict = {key:len(key.split()) for key in in_entity_freq_dict.keys()}
    cnt_len = 0
    for val in temp_entity_length_dict.values():
        cnt_len += val
    entity_density = cnt_len / doc_len
    entity_density_list.append(entity_density)

    # get a consistency of entity per document
    
    sentence = " ".join([word for word in words])
    for key in in_entity_freq_dict.keys():
        orig_key = copy.deepcopy(key)
        specialChars = '~`!@#$%^&*()_-+=[{]}:;\'",<.>/?|'
        for char in specialChars:
            key = key.replace(char, '\%c'%char)

        key_list = re.findall(key, sentence)
        if orig_key not in entity_cons_dict:
            entity_cons_dict[orig_key] = []

        entity_cons_dict[orig_key].append(in_entity_freq_dict[orig_key] / len(key_list))

    return entity_freq_dict, entity_density_list, entity_cons_dict

def entity_length(entity_freq_dict):
    entity_len_dict = {key:len(key.split()) for key in entity_freq_dict.keys()}
    return entity_len_dict

def document_length(doc_len, words):
    doc_len.append(len(words))
    return doc_len

def out_of_density(train_entity_freq_dict, test_entity_freq_dict, out_of_dens_dict, test_doc_len, test_data):
    train_entity_set = set([key for key in train_entity_freq_dict.keys()])
    test_entity_set = set([key for key in test_entity_freq_dict.keys()])

    diff_set = train_entity_set - test_entity_set

    for data_idx, data_inst in tqdm(enumerate(test_data), desc='Out of Density'):
        words = data_inst['str_words']
        labels = data_inst['tags']

        sentence = " ".join([word for word in words])

        for key in list(diff_set):
            orig_key = copy.deepcopy(key)
            specialChars = '~`!@#$%^&*()_-+=[{]}:;\'",<.>/?|'
            for char in specialChars:
                key = key.replace(char, '\%c'%char)

            key_list = re.findall(key, sentence)

            if key_list:
                if orig_key not in out_of_dens_dict:
                    out_of_dens_dict[orig_key] = []
                out_of_dens_dict[orig_key].append(len(orig_key.split()) / test_doc_len[data_idx])

    return out_of_dens_dict

def length_per_cons(train_entity_freq_dict, train_entity_cons_dict):
    length_cons_dict = {}
    for key, val in train_entity_freq_dict.items():
        if len(key.split()) not in length_cons_dict:
            length_cons_dict[len(key.split())] = []

    for key, val in train_entity_cons_dict.items():
        length_cons_dict[len(key.split())].append(np.mean(val))

    for key in sorted(length_cons_dict):
        print ("length:%d, mean consistency:%.4f" % (key, np.mean(length_cons_dict[key])))
    
    print ()
    return length_cons_dict

# def dens_per_cons(train_entity_freq_dict, train_entity_density_list):
#     dens_cons_dict = {}
#     import pdb; pdb.set_trace()
#     # for key, val in train_entity_freq_dict.items():
        
#     return dens_cons_dict

def main():
    data_dir = '../data/'
    entity_list = ['ncbi-disease']
    file_list = ['doc_train.json', 'doc_test.json']
    # file_list = ['doc_train.json', 'doc_dev.json']
    # entity_name = 'ncbi-disease' 
    for entity_name in entity_list:
        out_of_dens_dict = {}
        for file_name in file_list:
            with open(data_dir+'/'+entity_name+'/'+'from_rawdata'+'/'+file_name, 'r') as fp:
                data = json.load(fp)
                token_freq_dict, entity_freq_dict = {}, {}
                token_cons_dict, entity_cons_dict = {}, {}
                doc_len,entity_density_list = [], []

                for data_idx, data_inst in tqdm(enumerate(data), desc='Total Run'):
                    words = data_inst['str_words']
                    labels = data_inst['tags']
                    
                    token_freq_dict, token_cons_dict = token_frequency(token_freq_dict, token_cons_dict, words, labels, entity_name)
                    entity_freq_dict, entity_density_list, entity_cons_dict = entity_frequency(entity_freq_dict, entity_density_list, entity_cons_dict, words, labels, entity_name)
                    doc_len = document_length(doc_len, words)

                entity_len_dict = entity_length(entity_freq_dict)

            if 'train' in file_name:
                train_entity_freq_dict = copy.deepcopy(entity_freq_dict)
                train_entity_cons_dict = copy.deepcopy(entity_cons_dict)
                # train_entity_density_list = copy.deepcopy(entity_density_list)

            if 'dev' in file_name:
                dev_data = copy.deepcopy(data)
                dev_entity_freq_dict = copy.deepcopy(entity_freq_dict)
                dev_doc_len = copy.deepcopy(doc_len) 

            if 'test' in file_name:
                test_data = copy.deepcopy(data)
                test_entity_freq_dict = copy.deepcopy(entity_freq_dict)
                test_doc_len = copy.deepcopy(doc_len)

        length_cons_dict = length_per_cons(train_entity_freq_dict, train_entity_cons_dict)

        import pdb; pdb.set_trace()
        # dens_cons_dict = dens_per_cons(train_entity_freq_dict, train_entity_density_list)

        # # get out of density through a set of training entites 
        # out_of_dens_dict = out_of_density(train_entity_freq_dict, test_entity_freq_dict, out_of_dens_dict, test_doc_len, test_data)
        # get_list = []
        # for key, val_list in out_of_dens_dict.items():
        #     get_list.append(np.mean(val_list))

        # print (np.mean(get_list), np.std(get_list))

        

    ##########################################################################################################################################################################################################
    # data_dir = '../models/fine-tuned'
    # entity_list = ['gellus']
    # file_list = ['preds_doc_test.json']
    # # file_list = ['preds_sent_test.json']
    # models="biobert-base"
    # EP="30"
    # LR="3e-5"
    # ML="512"
    # WM="0"
    # DATA_TYPE="doc"
    
    # for entity_name in entity_list:
    #     out_of_dens_dict = {}
    #     for file_name in file_list:
    #         with open(data_dir+'/'+entity_name+'/'+models+"_"+"EP%s_"%EP+"LR%s_"%LR+"ML%s_"%ML+"WM%s_"%WM+DATA_TYPE+'/'+"checkpoint-best/"+file_name, 'r') as fp:
    #             data = json.load(fp)
    #             token_freq_dict, entity_freq_dict = {}, {}
    #             token_cons_dict, entity_cons_dict = {}, {}
    #             doc_len,entity_density_list = [], []

    #             for data_idx, data_inst in tqdm(enumerate(data), desc='Total Run'):
    #                 words = data_inst['str_words']
    #                 labels = data_inst['tags']
                    
    #                 token_freq_dict, token_cons_dict = token_frequency(token_freq_dict, token_cons_dict, words, labels, entity_name)
    #                 entity_freq_dict, entity_density_list, entity_cons_dict = entity_frequency(entity_freq_dict, entity_density_list, entity_cons_dict, words, labels, entity_name)
    #                 doc_len = document_length(doc_len, words)

    #             entity_len_dict = entity_length(entity_freq_dict)

    #         if 'train' in file_name:
    #             train_entity_freq_dict = copy.deepcopy(entity_freq_dict)
    #             train_entity_cons_dict = copy.deepcopy(entity_cons_dict)
    #             # train_entity_density_list = copy.deepcopy(entity_density_list)
    #             length_cons_dict = length_per_cons(train_entity_freq_dict, train_entity_cons_dict)

    #         if 'test' in file_name:
    #             test_data = copy.deepcopy(data)
    #             test_entity_freq_dict = copy.deepcopy(entity_freq_dict)
    #             test_doc_len = copy.deepcopy(doc_len)
    #             test_entity_cons_dict = copy.deepcopy(entity_cons_dict)
    #             preds_length_cons_dict = length_per_cons(test_entity_freq_dict, test_entity_cons_dict)

    #     token_list = []
    #     for token,token_freq in token_freq_dict.items():
    #         token_list.append(token_freq)
    #     print ("Token Frequency %.4f" % (np.mean(token_list)))

    #     entity_list = []
    #     for entity, entity_freq in entity_freq_dict.items():
    #         entity_list.append(entity_freq)
    #     print ("Entity Frequency %.4f" % (np.mean(entity_list)))

    #     token_o_list = []
    #     token_dise_list = []
    #     token_chem_list = []
    #     for token, token_cons_val in token_cons_dict.items():
    #         idx_token_o_list = []
    #         idx_token_ent_list = []
    #         idx_token_ent2_list = []
    #         for token_cons_val_idx in token_cons_val:
    #             idx_token_o_list.append(token_cons_val_idx[0])
    #             idx_token_ent_list.append(token_cons_val_idx[1])
    #             if 'bc5cdr' == entity_name:
    #                 idx_token_ent2_list.append(token_cons_val_idx[2])
            
    #         token_o_list.append(np.mean(idx_token_o_list))
    #         token_dise_list.append(np.mean(idx_token_ent_list))
    #         if 'bc5cdr' == entity_name:
    #             token_chem_list.append(np.mean(idx_token_ent2_list))

    #     print ("Token Consistency O label %.4f"%(np.mean(token_o_list)))
    #     print ("Token Consistency First Entity label %.4f"%(np.mean(token_dise_list)))
    #     if 'bc5cdr' == entity_name:
    #         print ("Token Consistency Second Entity label %.4f"%(np.mean(token_chem_list)))
            
    #     entity_cons_list = []
    #     for entity, entity_cons in entity_cons_dict.items():
    #         first_entity_cons_list = []
    #         for entity_cons_idx in entity_cons:
    #             first_entity_cons_list.append(entity_cons_idx)

    #         entity_cons_list.append(np.mean(first_entity_cons_list))
            
    #     print ("Entity Consistency O Label %.4f" % (1 - np.mean(entity_cons_list)))
    #     print ("Entity Consisteny of Entity Label %.4f" % (np.mean(entity_cons_list)))
        
    #     entity_length_list = []
    #     for entity, entity_len in entity_len_dict.items():
    #         entity_length_list.append(entity_len)
    #     print ("Entity Length %.4f" % (np.mean(entity_length_list)))

    #     for key, val in preds_length_cons_dict.items():
    #         print (key, np.mean(val))

    ##########################################################################################################################################################################################################
                        

                

if __name__ == "__main__":
    main()
