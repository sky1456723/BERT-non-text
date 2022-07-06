import sys
import os
import random
import numpy as np

import torch.utils.data

from ..preprocess import preprocess_seq_for_rnn, preprocess_seq_for_tfm, preprocess_label_for_tfm

class Homology_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for PLUS Homology training - make all pairs """
    def __init__(self, sequences, labels, cmaps, encoder, cfg, rnn=True, max_len=None):
        self.sequences = sequences
        self.labels = labels
        self.cmaps = cmaps
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)**2

    def __getitem__(self, k):
        n = len(self.sequences)
        i, j = k // n, k % n
        sequence0, sequence1 = self.sequences[i], self.sequences[j]
        similarity_level = self.labels[i, j]

        if self.rnn:
            instance0 = preprocess_seq_for_rnn(sequence0, self.num_alphabets, self.cfg)
            instance1 = preprocess_seq_for_rnn(sequence1, self.num_alphabets, self.cfg)
            if self.cmaps is not None: return instance0, instance1, similarity_level, self.cmaps[i], self.cmaps[j]
            else:                      return instance0, instance1, similarity_level
        else:
            instance = preprocess_seq_for_tfm(sequence0, sequence1, self.num_alphabets, self.cfg, self.max_len)
            if self.cmaps is not None: return (*instance, similarity_level, self.cmaps[i], self.cmaps[j])
            else:                      return (*instance, similarity_level)


class PairedHomology_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for PLUS Homology evaluation """
    def __init__(self, sequences0, sequences1, labels, cmaps0, cmaps1, encoder, cfg, rnn=False, max_len=None):
        self.sequences0 = sequences0
        self.sequences1 = sequences1
        self.labels = labels
        self.cmaps0 = cmaps0
        self.cmaps1 = cmaps1
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        self.augment = True
        if not self.rnn: self.set_max_len(max_len)

    def __len__(self):
        return len(self.sequences0)

    def __getitem__(self, i):
        sequence0, sequence1 = self.sequences0[i], self.sequences1[i]
        similarity_level = self.labels[i]
        if self.rnn:
            instance0 = preprocess_seq_for_rnn(sequence0, self.num_alphabets, self.cfg, self.augment)
            instance1 = preprocess_seq_for_rnn(sequence1, self.num_alphabets, self.cfg, self.augment)
            if self.cmaps0 is not None: return instance0, instance1, similarity_level, self.cmaps0[i], self.cmaps1[i]
            else:                       return instance0, instance1, similarity_level
        else:
            instance = preprocess_seq_for_tfm(sequence0, sequence1, self.num_alphabets, self.cfg, self.max_len, self.augment)
            if self.cmaps0 is None: return (*instance, similarity_level)
            else:                   return (*instance, similarity_level, self.cmaps0[i], self.cmaps1[i])

    def set_max_len(self, max_len):
        """ set max_len """
        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = 128
            for sequence0, sequence1 in zip(self.sequences0, self.sequences1):
                if len(sequence0) + len(sequence1) + 3 > self.max_len:
                    self.max_len = len(sequence0) + len(sequence1) + 3

    def set_augment(self, augment):
        """ set augmentation flag """
        self.augment = augment


class PLUS_Seq_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for PLUS single sequence task training and evaluation """
    def __init__(self, sequences, labels, encoder, cfg, rnn=False, max_len=None, truncate=True, augment=True):
        self.sequences = sequences
        self.labels = labels
        self.valids = None
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        self.truncate = truncate
        self.augment = augment
        if not self.rnn: self.set_max_len(max_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        if self.rnn:
            instance = preprocess_seq_for_rnn(self.sequences[i], self.num_alphabets, self.cfg, self.augment)
            return instance, self.labels[i]
        else:
            instance_seq = preprocess_seq_for_tfm(self.sequences[i], None, self.num_alphabets, self.cfg, self.max_len, self.augment)
            if self.valids is None:
                return (*instance_seq, self.labels[i])
            else:
                instance_label = preprocess_label_for_tfm(self.labels[i], self.valids[i], self.max_len)
                return (*instance_seq, *instance_label)

    def set_max_len(self, max_len):
        """ set max_len """
        if max_len is not None:
            self.max_len = max_len
            if not self.truncate:
                # split sequences/labels longer than max_len
                sequences, labels, valids, l = [], [], [], self.max_len - 2
                for i in range(len(self.sequences)):
                    seq, label = self.sequences[i], self.labels[i]
                    while len(seq) > self.max_len - 2:
                        sequences.append(seq[:l]);  seq = seq[l:]
                        labels.append(label[:l]);   label = label[l:]
                        valids.append(False)
                    sequences.append(seq); labels.append(label); valids.append(True)

                self.sequences = sequences
                self.labels = labels
                self.valids = valids
        else:
            self.max_len = 128
            for sequence in self.sequences:
                if len(sequence) > self.max_len:
                    self.max_len = len(sequence) + 2

    def set_augment(self, augment):
        """ set augmentation flag """
        self.augment = augment


class Seq_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for PLUS single sequence task training and evaluation """
    def __init__(self, sequences, labels, encoder, tokenizer, args, max_len=None, truncate=True, cache_dir = None, split = 'train'):
        self.sequences = sequences
        self.labels = labels
        self.valids = None
        self.num_alphabets = len(encoder)
        self.tokenizer = tokenizer
        #self.cfg = cfg
        #self.rnn = rnn
        self.truncate = truncate
        self.augment = 0.0
        self.args = args
        self.cache_dir = cache_dir
        self.split = split
        #if not self.rnn: self.set_max_len(max_len)
        self.set_max_len(max_len)
        self.cached = self.check_cached(split = split)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        #if self.rnn:
        #    instance = preprocess_seq_for_rnn(self.sequences[i], self.num_alphabets, self.cfg, self.augment)
        #    return instance, self.labels[i]
        #else:
        if not self.cached:
            instance_seq = self.preprocess(self.sequences[i], None, self.num_alphabets, self.max_len)
        else:
            instance_seq = (self.input_ids[i], self.token_type_ids[i], self.attention_mask[i])
        if self.valids is None:
            return (*instance_seq, self.labels[i])
        else:
            instance_label = self.preprocess_label(self.labels[i], self.valids[i], self.max_len)
            return (*instance_seq, *instance_label)
    
    def check_cached(self, split):
        input_ids_path = os.path.join(self.cache_dir, f'cached_{split}_input_ids_{self.args["task"]}_{self.args["model"]}_{self.max_len}.pkl')
        token_type_ids_path = os.path.join(self.cache_dir, f'cached_{split}_token_type_{self.args["task"]}_{self.args["model"]}_{self.max_len}.pkl')
        attention_mask_path = os.path.join(self.cache_dir, f'cached_{split}_att_mask_{self.args["task"]}_{self.args["model"]}_{self.max_len}.pkl')
        load_input = 0
        load_token_type = 0
        load_att_mask = 0
        if os.path.exists(input_ids_path):
            self.input_ids = torch.load(input_ids_path)
            load_input = 1
        if os.path.exists(token_type_ids_path):
            self.token_type_ids = torch.load(token_type_ids_path)
            load_token_type = 1
        if os.path.exists(attention_mask_path):
            self.attention_mask = torch.load(attention_mask_path)
            load_att_mask = 1
        if (load_input and load_token_type and load_att_mask):
            print("[mydataset.py] Cached input loaded.")
        else:
            print("[mydataset.py] Cached input not loaded. Need to do preprocess.")
        return (load_input and load_token_type and load_att_mask)



    
    def set_max_len(self, max_len):
        """ set max_len """
        if max_len is not None:
            self.max_len = max_len
            if not self.truncate:
                # split sequences/labels longer than max_len
                sequences, labels, valids, l = [], [], [], self.max_len - 2
                for i in range(len(self.sequences)):
                    seq, label = self.sequences[i], self.labels[i]
                    while len(seq) > self.max_len - 2:
                        sequences.append(seq[:l]);  seq = seq[l:]
                        labels.append(label[:l]);   label = label[l:]
                        valids.append(False)
                    sequences.append(seq); labels.append(label); valids.append(True)

                self.sequences = sequences
                self.labels = labels
                self.valids = valids
        else:
            self.max_len = 128
            for sequence in self.sequences:
                if len(sequence) > self.max_len:
                    self.max_len = len(sequence) + 2

    def set_augment(self, augment):
        """ set augmentation flag """
        self.augment = augment
        
    def truncate_seq_pair(self, x0, x1, max_len):
        """ clip sequences for the maximum length limitation """
        if x1 is not None:
            max_len -= 3
            while True:
                if len(x0) + len(x1) <= max_len: break
                elif len(x0) > len(x1): x0 = x0[:-1]
                else: x1 = x1[:-1]
        else:
            max_len -= 2
            x0 = x0[:max_len]
        return x0, x1

    def preprocess(self, x0, x1=None, num_alphabets=21, max_len=512):
        """ pre-processing steps for PLUS-TFM pre-training """
        num_alphabets = self.num_alphabets
        special_tokens = {"MASK": torch.tensor([self.tokenizer.mask_token_id], dtype=torch.long),
                          "CLS":  torch.tensor([self.tokenizer.cls_token_id], dtype=torch.long),
                          "SEP":  torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long)}
        tokens = torch.zeros(max_len, dtype=torch.long)
        segments = torch.zeros(max_len, dtype=torch.long)
        input_mask = torch.zeros(max_len, dtype=torch.long)

        # -3  for special tokens [CLS], [SEP], [SEP]
        x0, x1 = self.truncate_seq_pair(x0, x1, max_len)

        # set tokens and segments
        if x1 is not None:
            pair_len = len(x0) + len(x1) + 3
            tokens[:pair_len] = torch.cat([special_tokens["CLS"], x0, special_tokens["SEP"], x1, special_tokens["SEP"]])
            segments[len(x0) + 2:pair_len] = 1
            input_mask[:pair_len] = 1 # True
        else:
            single_len = len(x0) + 2
            tokens[:len(x0) + 2] = torch.cat([special_tokens["CLS"], x0, special_tokens["SEP"]])
            input_mask[:len(x0) + 2] = 1 # True

        #if self.augment == 0:
        #    return tokens, segments, input_mask

        if self.augment != 0:
            for pos in range(1, len(x0) + 1):
                if random.random() < self.augment: tokens[pos] = random.randint(1, num_alphabets - 1)

        return tokens, segments, input_mask
    def preprocess_label(self, y, v, max_len):
        """ pre-processing steps for PLUS-TFM fine-tuning """
        labels = torch.zeros(max_len, dtype=torch.long)
        valids = torch.zeros(1, dtype=torch.uint8)
        weights = torch.zeros(max_len, dtype=torch.bool)

        labels[1:len(y) + 1] = y
        valids[0] = 1 if v else 0
        weights[1:len(y) + 1] = True

        return labels, valids, weights


class Embedding_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for protein sequence embedding """
    def __init__(self, sequences, encoder, cfg, rnn=False):
        self.sequences = sequences
        self.valids = None
        self.num_alphabets = len(encoder)
        self.cfg = cfg
        self.rnn = rnn
        if not self.rnn: self.set_max_len()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        if self.rnn: instance = preprocess_seq_for_rnn(self.sequences[i], self.num_alphabets, self.cfg, augment=False)
        else:        instance = preprocess_seq_for_tfm(self.sequences[i], None, self.num_alphabets, self.cfg, self.max_len, augment=False)

        return instance

    def set_max_len(self):
        """ set max_len """
        self.max_len = 128
        for sequence in self.sequences:
            if len(sequence) > self.max_len:
                self.max_len = len(sequence) + 2
                
                
class HomolgySampler(torch.utils.data.sampler.Sampler):
    """ Weighted sampling of considering the similarity levels and their number of seq pairs """
    def __init__(self, labels, cfg):
        similarity = labels.numpy().sum(2)
        levels, counts = np.unique(similarity, return_counts=True)
        order = np.argsort(levels)
        levels, counts = levels[order], counts[order]
        weights = counts ** cfg.tau / counts
        weights = torch.as_tensor(weights, dtype=torch.double)

        similarity = similarity.ravel()
        levels, counts = np.unique(similarity, return_counts=True)
        order = np.argsort(levels)
        levels, counts = levels[order], counts[order]
        similarity_counts = np.zeros((len(levels) + 1), dtype=np.int32)
        for i in range(len(levels)):
            similarity_counts[i+1] = similarity_counts[i] + counts[i]
        similarity_order = np.argsort(similarity)

        self.weights = weights
        self.similarity_counts = similarity_counts
        self.similarity_order = similarity_order
        self.num_samples = cfg.epoch_size
        self.replacement = False

    def __iter__(self):
        level_sampling = torch.multinomial(self.weights, self.num_samples, replacement=True)
        sampled_levels, sampled_counts = np.unique(level_sampling, return_counts=True)

        sampled_pairs = []
        for l, c in zip(sampled_levels, sampled_counts):
            idxs = np.random.randint(0, self.similarity_counts[l]+1, c)
            idxs = self.similarity_order[idxs]
            sampled_pairs += idxs.tolist()

        return iter(sampled_pairs)

    def __len__(self):
        return self.num_samples
