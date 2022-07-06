import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
from tqdm import tqdm

from enum import Enum
import transformers as tr
from transformers.data.datasets import GlueDataset,GlueDataTrainingArguments
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from transformers.data.data_collator import DataCollator, default_data_collator
import copy
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str)
parser.add_argument('--task', type = str)
parser.add_argument('--mode', type = str, default = 'scratch pretrain')
parser.add_argument('--glue_dir', type = str, default = '/mnt/storage1/Dataset/GLUE')
parser.add_argument('--save_dir', type = str, default = './Analysis/perturbation')
parser.add_argument('--shift', type = int, default = 0)
parser.add_argument('--seed', type = str, default = '100')
parser.add_argument('--std', type = str, default = 1e-2)
args = parser.parse_args()

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

            config = tr.AutoConfig.from_pretrained(model_name, num_labels = 3)
            if mode == 'scratch':
                model = tr.AutoModel.from_config(config = config)
            else:
                model = tr.AutoModel.from_pretrained(model_name, config = config)
            model = model.to(device)
            model.train()
            print("OK")

            std = eval(args.std)
            
            dist = torch.distributions.normal.Normal(0,std)
            noise = {}
            difference = []

            torch.manual_seed(seed)
            for i, x in enumerate(tqdm(dataloader)):
                for name, param in model.named_parameters():
                    noise[name] = dist.sample(param.shape)
                    noise[name] = noise[name].to(device)


                x.pop('labels')
                with torch.no_grad():
                    for k in x.keys():
                        if isinstance(x[k], torch.Tensor):
                            x[k] = x[k].to(device)
                        if k=='input_ids':
                            x[k] = torch.remainder(x[k]+shift, model.config.vocab_size)
                    output1 = model(**x)[0][:,0,:].detach()
                    for name, param in model.named_parameters():
                        param.data = param.data + noise[name]

                    output2 = model(**x)[0][:,0,:].detach() 
                    difference.append(torch.norm(output1-output2, dim = -1).detach().cpu())


                    for name, param in model.named_parameters():
                        param.data = param.data - noise[name]
                if i >= 2: #if i >= 200:
                    break
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, mode+'_'+model_name+'_shift'+str(shift)+'_std'+str(std)+'_'+task_name+'_rand'+str(seed)+'.pkl')
            torch.save(difference, save_path)