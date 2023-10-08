from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import sys
from io import open
import xml.etree.ElementTree as ET
from collections import Counter
import pickle
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd

IGNORE_INDEX = -100

class InputExample_plus():
    def __init__(self,text,target,label,reasons,did = None,rank_scores = None):
        self.text_a = text
        self.text_b = target
        self.label = label
        self.rank_scores = rank_scores
        self.reasons = reasons
        self.id = did


class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label,did = None, scores = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.id = did
        self.scores = scores

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

class SentProcessorStance(DataProcessor):
    def get_train_examples(self, data_dir,ids, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='train', ids = ids,using_score=using_score)

    def get_dev_examples(self, data_dir, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='val',using_score=using_score)

    def get_test_examples(self, data_dir, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='test',using_score=using_score)

    def _create_examples(self, data_dir, set_type, ids = 0, using_score = True):
        stance_dic = {'FAVOR':0,'AGAINST':1,'NONE':2}
        input_examples = []
        if set_type == 'train':
            data = pd.read_csv(data_dir +'/train_rank.csv')
            if ids is not None:
                data = data.iloc[ids]
        else: 
            data = pd.read_csv(data_dir +'/raw_'+set_type+'_all_onecol.csv')

        if set_type == 'train':
            for i in range(len(data)):
                d = data.iloc[i]
                reasons = [d['reason 0'],d['reason 1'],d['reason 2']]
                if using_score:
                    scores = [0.0,0.0,0.0]
                    scores[stance_dic[d['Stance 1']]] = 1.0
                    input_examples.append(InputExample_plus(text = d['Tweet'],target = d['Target 1'], label = stance_dic[d['Stance 1']],reasons = reasons, rank_scores = scores,did = d['id']))
                else:
                    reason = reasons[stance_dic[d['Stance 1']]]
                    input_examples.append(InputExample_plus(text = d['Tweet'],target = d['Target 1'], label =  stance_dic[d['Stance 1']],reasons = reason, rank_scores = None,did = d['id']))
            
        else:
            for i in range(len(data)):
                d = data.iloc[i]
                input_examples.append(InputExample_plus(text = d['Tweet'],target = d['Target 1'], label = stance_dic[d['Stance 1']],reasons = None, rank_scores = None,did = d['id']))
        return input_examples

class SentProcessorSci(DataProcessor):
    def get_train_examples(self, data_dir,ids, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='train', ids = ids,using_score=using_score)

    def get_dev_examples(self, data_dir, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='dev',using_score=using_score)

    def get_test_examples(self, data_dir, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='test',using_score=using_score)

    def _create_examples(self, data_dir, set_type, ids = 0, using_score = True):
        stance_dic = {'Not Relevant':0,'Relevant':1}
        input_examples = []
        if set_type == 'train':
            data = pd.read_csv(data_dir +'/train_reasons.csv')
            if ids is not None:
                data = data.iloc[ids]
        else: 
            data = pd.read_csv(data_dir +'/'+set_type+'.csv')
        
        if set_type == 'train':
            for i in range(len(data)):
                d = data.iloc[i]
                reasons = [d['reason 1'],d['reason 2'],d['reason 3']]
                if using_score:
                    scores = [0.0,0.0,0.0]
                    scores[d['label']] = 1.0
                    input_examples.append(InputExample_plus(text = d['text'],target = d['topic'], label = d['label'],reasons = reasons, rank_scores = scores,did = d['id']))
                else:
                    reason = reasons[d['label']]
                    input_examples.append(InputExample_plus(text = d['text'],target = d['topic'], label = d['label'],reasons = reason, rank_scores = None,did = d['id']))

        else:
            for i in range(len(data)):
                d = data.iloc[i]
                input_examples.append(InputExample_plus(text = d['text'],target = d['topic'], label = d['label'],reasons = None, rank_scores = None,did = None))

        return input_examples

class SentProcessorRTE(DataProcessor):
    def get_train_examples(self, data_dir,ids, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='train', ids = ids, using_score = using_score)

    def get_dev_examples(self, data_dir,using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='dev', using_score = using_score)

    def get_test_examples(self, data_dir,using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='test', using_score = using_score)

    def _create_examples(self, data_dir, set_type, ids = 0, using_score = True):
        label_dic = {'not_entailment':0,'entailment':1}
        input_examples = []
        # print(max(ids))
        if set_type == 'train':
            data = pd.read_csv(data_dir +'/train_reasons.csv')
            if ids is not None:
                data = data.iloc[ids]
        else: 
            data = pd.read_csv(data_dir +'/'+set_type+'.csv')
        
        if set_type == 'train':
            for i in range(len(data)):
                d = data.iloc[i]
                reasons = [d['reason 1'],d['reason 2'],d['reason 3']]
                if using_score:
                    scores = [0.0,0.0,0.0]
                    scores[label_dic[d['label']]] = 1.0
                    input_examples.append(InputExample_plus(text = d['sentence1'],target = d['sentence2'], label = label_dic[d['label']],reasons = reasons, rank_scores = scores,did = d['id']))
                else:
                    reason = reasons[label_dic[d['label']]]
                    input_examples.append(InputExample_plus(text = d['sentence1'],target = d['sentence2'], label = label_dic[d['label']],reasons = reason, rank_scores = None,did = d['id']))

        else:
            for i in range(len(data)):
                d = data.iloc[i]
                input_examples.append(InputExample_plus(text = d['sentence1'],target = d['sentence2'], label = label_dic[d['label']],reasons = None, rank_scores = None,did = None))

        return input_examples

class SentProcessorMRPC(DataProcessor):
    def get_train_examples(self, data_dir,ids,using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='train', ids = ids, using_score = using_score)

    def get_dev_examples(self, data_dir,using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='dev', using_score = using_score)

    def get_test_examples(self, data_dir,using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='test', using_score = using_score)
        
    def _create_examples(self, data_dir, set_type, ids = 0, using_score = True):
        input_examples = []
        if set_type == 'train':
            data = pd.read_csv(data_dir +'/train_reasons.csv')
            if ids is not None:
                data = data.iloc[ids]
        else: 
            data = pd.read_csv(data_dir +'/'+set_type+'.csv')
        
        if set_type == 'train':
            for i in range(len(data)):
                d = data.iloc[i]
                reasons = [d['reason 1'],d['reason 2'],d['reason 3']]
                if using_score:
                    scores = [0.0,0.0,0.0]
                    scores[d['Quality']] = 1.0
                    input_examples.append(InputExample_plus(text = d['#1 String'],target = d['#2 String'], label = d['Quality'],reasons = reasons, rank_scores = scores,did = d['id']))
                else:
                    reason = reasons[d['Quality']]
                    input_examples.append(InputExample_plus(text = d['#1 String'],target = d['#2 String'], label = d['Quality'],reasons = reason, rank_scores = None,did = d['id']))

        else:
            for i in range(len(data)):
                d = data.iloc[i]
                input_examples.append(InputExample_plus(text = d['#1 String'],target = d['#2 String'], label = d['Quality'],reasons = None, rank_scores = None,did = None))

        return input_examples

class SentProcessorDEBA(DataProcessor):
    def get_train_examples(self, data_dir,ids, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='train', ids = ids, using_score = using_score)

    def get_dev_examples(self, data_dir, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='dev', using_score = using_score)

    def get_test_examples(self, data_dir, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='test', using_score = using_score)

    def _create_examples(self, data_dir, set_type, ids = 0, using_score = True):
        input_examples = []
        # 0- disagree, 1- neutral, 2- agree
        if set_type == 'train':
            # data = pd.read_csv(data_dir +'/train_reasons.csv')
            data = pd.read_csv(data_dir +'/train_reasons.csv')
            if ids is not None:
                data = data.iloc[ids]
        else: 
            data = pd.read_csv(data_dir +'/'+set_type+'.csv')
        
        if set_type == 'train':
            for i in range(len(data)):
                d = data.iloc[i]
                reasons = [d['reason 1'],d['reason 2'],d['reason 3']]
                if using_score:
                    scores = [0.0,0.0,0.0]
                    scores[d['label']] = 1.0
                    input_examples.append(InputExample_plus(text = d['body_parent'],target = d['body_child'], label = d['label'],reasons = reasons, rank_scores = scores,did = d['id']))
                else:
                    reason = reasons[d['label']]
                    input_examples.append(InputExample_plus(text = d['body_parent'],target = d['body_child'], label = d['label'],reasons = reason, rank_scores = None,did = d['id']))
        else:
            for i in range(len(data)):
                d = data.iloc[i]
                input_examples.append(InputExample_plus(text = d['body_parent'],target = d['body_child'], label = d['label'],reasons = None, rank_scores = None,did = None))
        return input_examples

class SentProcessorMAMS(DataProcessor):
    def get_train_examples(self, data_dir,ids, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='train', ids = ids, using_score = using_score)

    def get_dev_examples(self, data_dir, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='dev', using_score = using_score)

    def get_test_examples(self, data_dir, using_score = True):
        return self._create_examples(data_dir=data_dir, set_type='test', using_score = using_score)

    def _create_examples(self, data_dir, set_type, ids = 0, using_score = True):
        label_dic = {'negative':0,'positive':1,'neutral':2}
        input_examples = []
        if set_type == 'train':
            # data = pd.read_csv(data_dir +'/train_reasons.csv')
            data = pd.read_csv(data_dir +'/train_reasons.csv')
            if ids is not None:
                data = data.iloc[ids]
        else: 
            data = pd.read_csv(data_dir +'/'+set_type+'.csv')
        
        if set_type == 'train':
            for i in range(len(data)):
                d = data.iloc[i]
                reasons = [d['reason 1'],d['reason 2'],d['reason 3']]
                if using_score:
                    scores = [0.0,0.0,0.0]
                    scores[label_dic[d['label']]] = 1.0
                    input_examples.append(InputExample_plus(text = d['review'],target = d['category'], label = label_dic[d['label']],reasons = reasons, rank_scores = scores,did = d['id']))
                else:
                    reason = reasons[label_dic[d['label']]]
                    input_examples.append(InputExample_plus(text = d['review'],target = d['category'], label = label_dic[d['label']], reasons = reason, rank_scores = None,did = d['id']))
        else:
            for i in range(len(data)):
                d = data.iloc[i]
                input_examples.append(InputExample_plus(text = d['review'],target = d['category'], label = label_dic[d['label']],reasons = None, rank_scores = None,did = None))
        return input_examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_a.pop()

def convert_exp_reas_to_features_(examples, max_seq_length,
                                 tokenizer,input_ = False, max_reason = 60):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if input_:
            text =  '<cls> '+example.text_a + ' <extra_id_0> ' + example.text_b 
            inputs = tokenizer(text, padding='max_length', max_length=max_seq_length,truncation=True)
        else:
            text =  example.reasons
            inputs = tokenizer(text, padding='max_length', max_length=max_reason,truncation=True)

 
        features.append(
                InputFeature(input_ids=inputs['input_ids'],
                              input_mask=inputs['attention_mask'],
                              segment_ids=None,
                              label=example.label,
                              did = example.id,
                              scores=example.rank_scores
                              ))
    return features

def convert_exp_reas_to_features_modify(examples, max_seq_length,
                                 tokenizer,input_ = False, max_reason = 60):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if input_:
            text_a = tokenizer.tokenize(example.text_a)
            text_b = tokenizer.tokenize(example.text_b)
            if len(text_a)+len(text_b)>max_seq_length-4:
                text_a = tokenizer.convert_tokens_to_string(text_a[:max_seq_length-4-len(text_b)])
            else: text_a = example.text_a
            text =  ' <cls> ' + text_a + ' <extra_id_0> ' + example.text_b 
            inputs = tokenizer(text, padding='max_length', max_length=max_seq_length,truncation=True)
        else:
            text =  example.reasons
            inputs = tokenizer(text, padding='max_length', max_length=max_reason,truncation=True)

        features.append(
                InputFeature(input_ids=inputs['input_ids'],
                              input_mask=inputs['attention_mask'],
                              segment_ids=None,
                              label=example.label,
                              did = example.id,
                              scores=example.rank_scores
                              ))
    return features
