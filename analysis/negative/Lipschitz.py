import torch
from tqdm import tqdm
import argparse

import os
import numpy as np
import pandas as pd
import scipy

from enum import Enum
import transformers as tr
from transformers.data.datasets import GlueDataset,GlueDataTrainingArguments
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from transformers.data.data_collator import DataCollator, default_data_collator
import copy
device = 'cuda:0'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str)
parser.add_argument('--task', type = str)
parser.add_argument('--mode', type = str, default = 'scratch pretrain')
parser.add_argument('--glue_dir', type = str, default = '/mnt/storage1/Dataset/GLUE')
parser.add_argument('--save_dir', type = str, default = './Analysis/test')
parser.add_argument('--shift', type = int, default = 0)
parser.add_argument('--seed', type = str ,default = '42')
args = parser.parse_args()

model_name = args.model

#l_dict = {}
#config = transformers.AutoConfig.from_pretrained(model_name, output_preLN = True)
#model = transformers.AutoModel.from_pretrained(model_name, 
#                                                             config = config).to(device)
#for name, param in model.bert.named_parameters():
#    l_dict[name] = []
#for f in tqdm(files):
    #state_dict = torch.load(os.path.join('/mnt/storage2/pretrain_ckpt/bert', f))
    #model.load_state_dict(state_dict)
'''
for name, param in model.bert.named_parameters():
    if 'embeddings' in name:
        continue
    if len(param.shape) == 2:
        u, s, v = torch.svd(param, compute_uv = False)
        l_dict[name].append(torch.max(torch.abs(s)).item())
    else:
        l_dict[name].append(torch.max(torch.abs(param)).item())
'''     
for iii ,task_name in enumerate(args.task.split(" ")):
    for seed in args.seed.split(" "):
        torch.manual_seed(seed)
        
        data_dir = args.glue_dir
        model_name = args.model
        data_args = tr.GlueDataTrainingArguments(task_name = task_name,
                                                 data_dir = os.path.join(data_dir, task_name),
                                                 )
        tokenizer = tr.AutoTokenizer.from_pretrained(model_name)


        dataset = tr.GlueDataset(data_args, tokenizer=tokenizer)
        sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=32, 
                                                 sampler = sampler,
                                                 collate_fn=default_data_collator,
                                                 drop_last=False)
        for mode in args.mode.split(" "):
            
            shift = args.shift
            torch.manual_seed(seed)

            config = tr.AutoConfig.from_pretrained(model_name)
            config.output_preLN = True
            if mode == 'scratch':
                model = tr.AutoModel.from_config(config = config)
            else:
                model = tr.AutoModel.from_pretrained(model_name, config = config)
            model = model.to(device)
            model.train()
            print("OK")

            std = {}
            for n in range(config.num_hidden_layers):
                std[n] = {'FFN':[], 'Attn':[]}

            torch.manual_seed(seed)
            for i, x in enumerate(tqdm(dataloader)):

                x.pop('labels')
                with torch.no_grad():
                    for k in x.keys():
                        if isinstance(x[k], torch.Tensor):
                            x[k] = x[k].to(device)
                        if k=='input_ids':
                            x[k] = torch.remainder(x[k]+shift, model.config.vocab_size)
                    output = model(**x)
                    preLN = output[-1]
                    for n in range(config.num_hidden_layers):
                        std[n]['FFN'].append(torch.std(preLN[n][0], dim = -1).view(-1).cpu())
                        std[n]['Attn'].append(torch.std(preLN[n][1], dim = -1).view(-1).cpu())
                if i >= 2: #if i >= 200:
                    for n in range(config.num_hidden_layers):
                        std[n]['FFN'] = torch.cat(std[n]['FFN'])
                        std[n]['Attn'] = torch.cat(std[n]['Attn'])
                    break
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir,
                                     mode+'_'+model_name+'_shift'+str(shift)+'_'+task_name+'_rand'+str(seed)+'.pkl')
            torch.save(std, save_path)
            if mode == 'scratch':
                torch.save(model.state_dict(), os.path.join(args.save_dir, mode+'_'+model_name+'_shift'+str(shift)+'_'+task_name+'_rand'+str(seed)+'_model.pkl'))
