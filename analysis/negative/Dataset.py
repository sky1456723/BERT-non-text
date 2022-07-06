import torch
import torch.utils.data
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import XLNetTokenizer
from transformers import DistilBertTokenizer
from transformers import AlbertTokenizer
from transformers import RobertaTokenizer
from transformers import AutoTokenizer
import torch.nn.functional as F
import pandas as pd
import numpy as np

import random
import copy
import math
from tqdm import tqdm

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

class MaskedLMDataset(torch.utils.data.Dataset):
    def __init__(self, mask_id = 103, mask_ratio = 0.15):
        super(MaskedLMDataset, self).__init__()
        self.mask_id = mask_id
        self.mask_ratio = mask_ratio
    def __getitem__(self):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def mask_sentence(self, sentence, max_token_id = 28996):
        '''
        sentence: a list of token id
        '''
        
        sentence_len = len(sentence)
        rand_num = random.random()
        size = max([1, int(math.ceil(sentence_len*self.mask_ratio))])
        
        index = list(range(0, sentence_len))
        np.random.shuffle(index)
        mask_position = index[:size]
        if rand_num <= 0.8:
            '''
            Do mask
            '''
            for i in mask_position:
                sentence[i] = self.mask_id
        elif rand_num <= 0.9:
            '''
            Do random replace
            '''
            rand_word = list(np.random.randint(low = 0, high = max_token_id, size= size))
            for i in range(len(mask_position)):
                sentence[mask_position[i]] = rand_word[i]
        return sentence, mask_position
    

    
class Dataset_bert(MaskedLMDataset):
    def __init__(self, 
                 data_path, 
                 label_path = None, 
                 tokenizer_type='bert-base-uncased', 
                 do_lower_case = False, 
                 sent_num = 1,
                 size = -1,
                 mask = False, 
                 mask_ratio = 0.15, 
                 lang = None,
                 cross_lingual = False,
                 target_lang = None):
        super(Dataset_bert, self).__init__(mask_ratio = mask_ratio)
        self.data_file = open(data_path)
        self.mask = mask
        if label_path:
            self.label_file = open(label_path)
        self.label = []
        self.data = []
        self.tokenized_sentences = []
        self.sentence_lens = []
        self.segment_ids = []
        
        self.set_tokenizer(tokenizer_type, do_lower_case)
        self.lang = lang
        self.cross_lingual = cross_lingual
        if cross_lingual:
            self.max_index = AutoTokenizer.from_pretrained(target_lang, 
                                                           do_lower_case = do_lower_case).vocab_size
        
        file_type = data_path.split('.')[-1]
        self.load_data_to_str(file = file_type, size = size)
        if sent_num == 1:
            self.process_single()
        else:
            self.process_pair()
            
        
            
    def __len__(self):
        '''
        return how many sentences are in this dataset 
        '''
        return len(self.tokenized_sentences)
    def __getitem__(self, index):
        '''
        With prob. 0.5, the dataset will return a wrong pair (randomly choose answer,
        generated by the function "gen_wrong_pair"), with prob. 0.5, generate a correct pair.
        
        one data consisting of four term:
        1. data: shape (sentence_length, feature_dim)
        2. segment_id: to tell bert which word belongs to the first/second sentence
        3. sentence_lens: the length of the pair of sentence before padding
        4. label: 1 means correct, 0 means wrong
        '''
        label = 1
        if self.mask:
            clean_sent = self.tokenized_sentences[index]
            to_mask_sent = copy.deepcopy(self.tokenized_sentences[index])
            sent, mask_position = self.mask_sentence(to_mask_sent, max_token_id = self.tokenizer.vocab_size)
        
        if len(self.label)>0:
            if self.mask:
                return [sent, self.segment_ids[index], self.sentence_lens[index], self.label[index], mask_position, clean_sent]
            else:
                return [self.tokenized_sentences[index], self.segment_ids[index], self.sentence_lens[index], self.label[index]]
        else:
            if self.mask:
                return [sent, self.segment_ids[index], self.sentence_lens[index], 0, mask_position, clean_sent]
            else:
                return [self.tokenized_sentences[index], self.segment_ids[index], self.sentence_lens[index], -1]
    def set_tokenizer(self, tokenizer, do_lower_case = False):
        if type(tokenizer)==str:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, 
                                                           do_lower_case = do_lower_case)
        else:
            self.tokenizer = tokenizer
        print(self.tokenizer)
        self.mask_id = self.tokenizer.mask_token_id
                
    def load_data_to_str(self, size=-1, file = 'txt'):
        '''
        loading the dialogue data from txt file.
        size : means that how many sentences you want to load in this dataset,
               for example, user can first load 0~99 line of sentences(50 pair), and then load
               100~299 line(100 pairs, clean the sentences 0~99 before loading) to the dataset.
               If reading EOF, it will start at 0 line and continue loading.
               default: -1: loading all the data.
        
        '''
        if file == 'tsv':
            self.data = pd.read_csv(self.data_file, sep='\t')
            self.data.fillna('', inplace = True)
        elif file == 'csv':
            self.data = pd.read_csv(self.data_file)
            self.data.fillna('', inplace = True)
        elif file == 'txt':
            #self.data_file = open(self.data_file)
            self.data = []
            print(size)
            if size == -1:
                self.data = self.data_file.readlines()
                for i in range(len(self.data)):
                    self.data[i] = (self.data[i]).replace('\n','')
                print("Reading all data")
            else:
                for i in range(size):
                    try:
                        self.data.append( next(self.data_file).replace('\n',''))
                    except:
                        print("Read to the EOF, seek to the start")
                        self.data_file.seek(0)
                        self.data.append( next(self.data_file).replace('\n',''))
                print("Reading %d data" % size)   
                
    def process_single(self):
        new_label = []
        if hasattr(self, 'label_file'):
            self.label = self.label_file.readlines()
        for i in tqdm(range(len(self.data['sentence']))):
            encode_dict = self.tokenizer.encode_plus(self.data['sentence'][i])
            concat_sent = encode_dict['input_ids']
            exceed = False
            if self.cross_lingual:
                for w, word in enumerate(concat_sent):
                    if word > self.max_index:
                        concat_sent[w] = self.tokenizer.unk_token_id
            
            
            if 'token_type_ids' in encode_dict.keys():
                seg_id = encode_dict['token_type_ids']
            else:
                seg_id = [0]*len(concat_sent)
            
            self.tokenized_sentences.append( concat_sent)
            self.segment_ids.append(seg_id)
            self.sentence_lens.append(len(concat_sent))
            if hasattr(self, 'label_file'):
                self.label[i] = self.label[i].replace('\n', '')
                new_label.append(int(self.label[i]))
            elif 'label' in self.data.keys():
                new_label.append(self.data['label'][i])
        if len(new_label) > 0:
            self.label = new_label
        self.data_file.close()
        
        #print("Skip: ", skip_num)
        print("File closed")
    def process_pair(self):
        '''
        Process sentences to pair.
        It will skip and pop the sentences if the concatenated pair is too long (> 512) for BERT.
        '''
        first_utter = self.data[::2]
        second_utter = self.data[1::2]
        new_label = []
        if hasattr(self, 'label_file'):
            self.label = self.label_file.readlines()
        for i in tqdm(range(0,len(first_utter))):
            encode_dict = self.tokenizer.encode_plus(first_utter[i], second_utter[i])
            concat_sent = encode_dict['input_ids']
            if self.cross_lingual:
                for w, word in enumerate(concat_sent):
                    if word > self.max_index:
                        concat_sent[w] = self.tokenizer.unk_token_id
            
            
            if 'token_type_ids' in encode_dict.keys():
                seg_id = encode_dict['token_type_ids']
            else:
                seg_id = [0]*len(concat_sent)
            if len(concat_sent) > 384:
                print("Skip too long, index: ", i)
                continue
            
            self.sentence_lens.append(len(concat_sent))
            self.tokenized_sentences.append(concat_sent)
            self.segment_ids.append(seg_id)
            if hasattr(self, 'label_file'):
                self.label[i] = self.label[i].replace('\n', '')
                new_label.append(int(self.label[i]))
        self.label = new_label
            
        
        print("Process %d pairs of data" % (len(self.data)//2))
        #print("Skip: ", skip_num)
        self.data_file.close()
    
        



#PAD_id = BertTokenizer.from_pretrained('bert-base-uncased').convert_tokens_to_ids(['[PAD]'])[0]
def bert_collate_fn(input_list):
    '''
    This function should be passed to the dataloader.
    
    Generate batch data.
    It will process padding in one batch.
    "attention mask" tells BERT the padding shouldn't be calculated.
    output shape:
    1. data: (batch, max_sentence_length_in_this_batch)
    2. seg: (batch, max_sentence_length_in_this_batch)
    3. attention_mask: (batch, max_sentence_length_in_this_batch)
    4. sentence_len: (batch, 1)
    5. label: (batch, 1)
    '''
    data_seg_len = list(zip(*input_list))
    data = copy.deepcopy(data_seg_len[0])
    seg = copy.deepcopy(data_seg_len[1])
    if len(data_seg_len) >= 6:
        mask_id = copy.deepcopy(data_seg_len[4])
        clean_data = copy.deepcopy(data_seg_len[5])
        mask_id_max_len = max([len(i) for i in data_seg_len[4]])
    attention_mask = []
    max_len = max(data_seg_len[2])
    for i in range(len(data)):
        to_add = [1]*data_seg_len[2][i]
        to_add.extend([0]*(max_len - data_seg_len[2][i]))
        attention_mask.append(to_add)
        seg[i].extend([0]*(max_len - data_seg_len[2][i]))
        data[i].extend([0]*(max_len - data_seg_len[2][i]))
        if len(data_seg_len) >= 6:
            mask_id[i].extend([-1]*(mask_id_max_len - len(mask_id[i])))
            clean_data[i].extend([0]*(max_len - data_seg_len[2][i]))
            
    data = torch.LongTensor(data)
    seg = torch.LongTensor(seg)
    attention_mask = torch.LongTensor(attention_mask)
    sentence_len = torch.LongTensor(data_seg_len[2]).unsqueeze(dim=1)
    label = torch.Tensor(data_seg_len[3]).unsqueeze(dim=1)
    if len(data_seg_len) >= 6:
        clean_data = torch.LongTensor(clean_data)
        mask_id = torch.LongTensor(mask_id)
        return data, seg, attention_mask, sentence_len, label, mask_id, clean_data
    else:
        return data, seg, attention_mask, sentence_len, label


def lstm_collate_fn(input_list):
    '''
    This function should be passed to the dataloader.
    
    Generate batch data.
    It will process padding in one batch.
    "attention mask" tells BERT the padding shouldn't be calculated.
    output shape:
    1. data: (batch, max_sentence_length_in_this_batch)
    2. seg: (batch, max_sentence_length_in_this_batch)
    3. attention_mask: (batch, max_sentence_length_in_this_batch)
    4. sentence_len: (batch, 1)
    5. label: (batch, 1)
    '''
    input_list.sort(key = lambda x:x[2], reverse = True)
    data_seg_len = list(zip(*input_list))
    data = copy.deepcopy(data_seg_len[0])
    max_len = max(data_seg_len[2])
    for i in range(len(data)):
        to_add = [1]*data_seg_len[2][i]
        to_add.extend([0]*(max_len - data_seg_len[2][i]))
        data[i].extend([0]*(max_len - data_seg_len[2][i]))
    data = torch.LongTensor(data)
    sentence_len = torch.LongTensor(data_seg_len[2])
    #data = torch.nn.utils.rnn.pack_padded_sequence(data, lengths = sentence_len, batch_first = True)
    #sentence_len = sentence_len.unsqueeze(dim = 1)
    label = torch.Tensor(data_seg_len[3])
    return data, sentence_len, label