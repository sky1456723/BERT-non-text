import torch
import numpy as np
import transformers
import random
import argparse
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str)
parser.add_argument('--data_dir', type = str, default = './Hilbert-CNN/data')
parser.add_argument('--save_dir', type = str, default = './data')
parser.add_argument('--model', type = str)
parser.add_argument('--seed', type = int, default = 100)
parser.add_argument('--ratio', type = float, default = 1.0)
args = parser.parse_args()
#args = vars(args)
random.seed(args.seed)
data_path = os.path.join(args.save_dir, args.task)

if not os.path.exists(data_path):
    os.makedirs(data_path)

#if not 'albert' in args.model:
#    tokenizer = transformers.BertTokenizerFast.from_pretrained(args.model)
#else:
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
tokenizer.model_max_length = 502

data_map = {'A':1, 'T':2, 'C':3, 'G':4}
data = []
label = []
attention_mask = []
cls_id = tokenizer.cls_token_id
sep_id = tokenizer.sep_token_id
print(cls_id, sep_id)
with open(os.path.join(args.data_dir, f'{args.task}.txt')) as files:
    text = files.readlines()
    for i in tqdm(range(len(text))):
        if i%3 == 0:
            continue
        elif i%3 == 1:
            raw_text = text[i].replace('\n', '')
            if not 'albert' in args.model:
                seq = " "
                seq = seq.join(text[i].replace('\n', ''))
            else:
                seq = ''
                seq = seq.join(text[i].replace('\n', ''))

            input_ids = tokenizer.encode_plus(seq, padding='max_length')
            if not 'chinese' in args.model:
                data.append(input_ids['input_ids'])
            else:
                raw = [cls_id]
                raw.extend([data_map[r] for r in raw_text])
                raw.append(sep_id)
                raw.extend([tokenizer.pad_token_id]*(502-len(raw)))
                data.append(raw)
            attention_mask.append(input_ids['attention_mask'])
        else:
            label.append(int(text[i]))

all_feature = list(zip(data, attention_mask, label))
random.shuffle(all_feature)
data, attention_mask, label = list(zip(*all_feature))
data = torch.LongTensor(data)
label = torch.LongTensor(label)
attention_mask = torch.Tensor(attention_mask)
total_len = int(data.shape[0]*args.ratio)
data = data[:total_len,:]
label = label[:total_len]
attention_mask = attention_mask[:total_len,:]
print(data.shape)
torch.save(data, os.path.join(data_path, f'{args.task}_{args.model}_data.pkl'))
torch.save(attention_mask, os.path.join(data_path, f'{args.task}_{args.model}_attention_mask.pkl'))
torch.save(label, os.path.join(data_path, f'{args.task}_{args.model}_label.pkl'))

