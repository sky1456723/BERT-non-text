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
parser.add_argument('--save_dir', type = str, default = './Analysis/grad_conf_new')
parser.add_argument('--shift', type = int, default = 0)
parser.add_argument('--seed', type = str, default = '100')
parser.add_argument('--accumulation', '-a', type = int, default = 1)
parser.add_argument('--op', type = str, default = 'cosine')
args = parser.parse_args()

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}



dimension = -1

def L2_dist(a, b, dim = dimension):
    return torch.norm(a - b, dim = dim)

if args.op == 'cosine':
    operation = F.cosine_similarity
elif args.op == 'dot':
    operation = torch.tensordot
elif args.op == 'l2':
    operation = L2_dist


for iii ,task_name in enumerate(args.task.split(" ")):
    for seed in args.seed.split(" "):
        torch.manual_seed(args.seed)
        #data_dir = '/mnt/storage1/Dataset/GLUE'
        model_name = args.model
        data_args = tr.GlueDataTrainingArguments(task_name = task_name,
                                                 data_dir = os.path.join(args.glue_dir, task_name),
                                                 )
        tokenizer = tr.AutoTokenizer.from_pretrained(model_name)


        dataset = tr.GlueDataset(data_args, tokenizer=tokenizer)
        sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=int(32/args.accumulation), 
                                                 sampler = sampler,
                                                 collate_fn=default_data_collator,
                                                 drop_last=False)



        for m in args.mode.split(" "):
            mode = m
            #shift = 1000

            config = tr.AutoConfig.from_pretrained(model_name, num_labels = num_labels[task_name.lower()])
            if mode == 'scratch':
                model = tr.AutoModelForSequenceClassification.from_config(config = config)
            else:
                model = tr.AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                              config = config)
            model = model.to(device)
            model.train()
            print("OK")
            accumulation_list = 0
            grad_list = []
            cos = []
            torch.manual_seed(int(seed))
            for i, x in enumerate(tqdm(dataloader)):
                for k in x.keys():
                    if isinstance(x[k], torch.Tensor):
                        x[k] = x[k].to(device)
                    if k=='input_ids':
                        x[k] = torch.remainder(x[k]+args.shift, model.config.vocab_size)
                out = model(**x)
                out[0].backward()
                with torch.no_grad():
                    grad = []
                    for name, param in model.named_parameters():
                        grad.append(param.grad.view(-1).cpu())
                    grad = torch.cat(grad)
                    #for name, param in model.named_parameters():
                    accumulation_list = accumulation_list + grad
                    if (i+1)%args.accumulation == 0:
                        #for name, param in model.named_parameters():
                        grad_list.append(accumulation_list/args.accumulation)
                        accumulation_list = 0
                model.zero_grad()
                if i/args.accumulation >= 2: #if i/args.accumulation >= 100:
                    break

            model = model.cpu()
            torch.cuda.empty_cache()
            del model
            
            '''
            for k in tqdm(grad_list.keys()):
                cos[k] = []
                #if not 'embedding' in k:
                for t in range(len(grad_list[k])):
                    grad_list[k][t] = grad_list[k][t].view(-1).to(device)
                for t in range(len(grad_list[k])):
                    for t2 in range(t+1, len(grad_list[k])):
                        print(grad_list[k][t].shape)
                        if args.op == 'dot':
                            cos[k].append(operation(grad_list[k][t], 
                                                    grad_list[k][t2], dims =1).item())
                        else:
                            cos[k].append(operation(grad_list[k][t], 
                                                    grad_list[k][t2], dim =0).item())
                for t in range(len(grad_list[k])):
                    #grad_list[k][t] = grad_list[k][t].cpu()
                    grad_list[k][t] = 0
            '''
            for t in tqdm(range(len(grad_list))):
                for t2 in range(t+1, len(grad_list)):
                    if args.op == 'dot':
                        cos.append(operation(grad_list[t], 
                                             grad_list[t2], dims =1).item())
                    else:
                        cos.append(operation(grad_list[t], 
                                             grad_list[t2], dim =0).item())
                grad_list[t] = 0
            
            #for k in cos.keys():
            #    cos[k] = np.array(cos[k])
            os.makedirs(args.save_dir, exist_ok = True)
            torch.save(cos, 
                       os.path.join(args.save_dir, args.op+'_'+args.model+'_'+mode+\
                       '_shift'+str(args.shift)+'_'+task_name+'_rand'+seed+'.pkl'))
