import torch
from Dataset import *
from tqdm import tqdm
import argparse

import os
import numpy as np
import pandas as pd
import scipy

from enum import Enum
import transformers
from transformers.data.datasets import GlueDataset,GlueDataTrainingArguments
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from transformers.data.data_collator import DataCollator, default_data_collator
import copy
device = 'cuda:0'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str)
parser.add_argument('--task', type = str)
parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--scratch', action = 'store_true')
parser.add_argument('--save_dir', type = str, default = './dynamical')
parser.add_argument('--glue_dir', type = str)
parser.add_argument('--shift', type = int, default = 0)
parser.add_argument('--seed', type = int ,default = 42)
args = parser.parse_args()

torch.manual_seed(args.seed)
class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

def batch_jacobian_singualr(last_layer, input_tensor, sent_len):
    to_dev = torch.sum(last_layer, dim = 0)[0]
    #to_dev = last_layer#.view(-1)
    grad_list = []
    for j in tqdm(range(to_dev.shape[0])):
        if j+1==to_dev.shape[0]:
            grad = torch.autograd.grad(to_dev[j], input_tensor)[0].cpu()
        else:
            grad = torch.autograd.grad(to_dev[j], input_tensor, retain_graph=True)[0].cpu()
        grad_list.append(grad)
    #to_dev = to_dev.detach()
    #input_tensor = input_tensor.detach()
    #del to_dev, input_tensor
    grad_list = torch.stack(grad_list, dim = 1)
    ans_list = []
    with torch.no_grad():
        for j in range(grad_list.shape[0]):
            u, s, v = torch.svd(grad_list[j, :, :sent_len[j], :].view(grad_list.shape[1], -1), 
                            compute_uv = False) 
            ans_list.append(s.cpu())
    del grad_list
    return ans_list


def sample_word_j_singualr(last_layer, input_tensor, seq_len, sample_rate = 1):
    # here assume batch size = 1
    ans_list = []
    
    #last_layer = torch.sum(last_layer, dim = 0)
    last_layer = last_layer[0]
    #input_tensor = input_tensor[0]
    if type(sample_rate)==float:
        sample = seq_len*sample_rate
    elif type(sample_rate)==int:
        sample = sample_rate
    
    if sample < 1:
        sample = 1
    else:
        sample = int(sample)
    
    
    #to_dev = last_layer#.view(-1)
    sample_last_layer = np.random.choice(seq_len, sample, replace = False)
    sample_first_layer = np.random.choice(seq_len, sample, replace = False)
    
    #grad_list = []
    
    
    for i in range(len(sample_last_layer)):
        to_dev = last_layer[ sample_last_layer[i] ]
        jacobian = [ [] for  ii in range(seq_len) ]
        
        for one_d in range(to_dev.shape[0]):
            #vec = torch.zeros_like(to_dev)
            #vec[j] = 1
            
            if one_d+1==to_dev.shape[0] and i+1==len(sample_last_layer):
                grad = torch.autograd.grad(to_dev[one_d], input_tensor)[0]#.cpu()
            else:
                grad = torch.autograd.grad(to_dev[one_d], input_tensor, 
                                           retain_graph=True)[0]#.cpu()
            
            for one_word in range(seq_len):
                jacobian[one_word].append(grad[0, one_word ])
        
        for one_w in range(seq_len):
            jacobian[one_w] = torch.stack(jacobian[one_w], dim = 0)
        jacobian = torch.stack(jacobian, dim = 0)
        with torch.no_grad():
            
            u, s, v = torch.svd(jacobian, 
                                compute_uv = False) 
            #print(s.shape)
            ans_list.append(s.cpu())
        del jacobian
    return ans_list

model_name = args.model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
task_name = args.task
data_dir = os.path.join(args.glue_dir, task_name)
#cache_dir = os.path.join(data_dir, 'cache_dev_{}_128_{}'.format(type(tokenizer).__name__, task_name.lower()))

data_args = GlueDataTrainingArguments(task_name = task_name, 
                                      data_dir = data_dir,
                                      max_seq_length=128)

dataset = GlueDataset(args = data_args, 
                      tokenizer = tokenizer,
                      mode = Split.train,
                      cache_dir = data_dir)
print('finish dataset')

sampler = RandomSampler(dataset)
dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1,
            collate_fn=default_data_collator,
            drop_last=False
        )

pad_index = tokenizer.pad_token_id
print('finish dataloader')

config = transformers.AutoConfig.from_pretrained(model_name, output_hidden_states = True)
if args.pretrain:
    mode = 'pretrain'
    model = transformers.AutoModel.from_pretrained(model_name, config = config)
elif args.scratch:
    mode = 'scratch'
    model = transformers.AutoModel.from_config(config = config)
model.to(device)

torch.manual_seed(args.seed)

ans_list = []
for index, x in enumerate(tqdm(dataloader)):
    x.pop('labels')
    seq_len = torch.sum(x['input_ids']!=pad_index).item()
    for k in x.keys():
        if isinstance(x[k], torch.Tensor):
            x[k] = x[k][:,:seq_len].to(device)
    output = model(**x)  
    ans_list.extend(sample_word_j_singualr(last_layer=output[0], input_tensor=output[2][0], seq_len = seq_len))

    if index+1 >= 2: #if index+1 >= 200:
        break
    
ans_list = torch.cat(ans_list, dim = 0)
os.makedirs(args.save_dir, exist_ok=True)
torch.save(ans_list, os.path.join(args.save_dir, mode+'_'+args.model+'_'+task_name+'_shift'+str(args.shift)+'.pkl'))
