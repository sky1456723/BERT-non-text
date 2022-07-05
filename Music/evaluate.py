import os
import scipy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np
import pandas as pd

import argparse
import tensorboardX

import transformers
from transformers.optimization import get_linear_schedule_with_warmup 
from tqdm import tqdm
import time
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str)
parser.add_argument('--model', type = str, default = 'bert-base-uncased')
parser.add_argument('--seed', type = int, default = 2020)
parser.add_argument('--type', type = str, choices=['pretrain', 'scratch'])
parser.add_argument('--split', type = str, choices=['dev', 'test'], default='dev')
parser.add_argument('--state_dict', type = str)
parser.add_argument('--batch_size', '-b', type = int, default = 32)
parser.add_argument('--shift_table', type = str, default = '')
parser.add_argument('--step', type = int, default=-1)

parser.add_argument('--logdir', type = str, default = './log')
parser.add_argument('--datadir', type = str)
args = vars(parser.parse_args())

#if args['filename'] == None:
#    args['filename'] = f'{args["task"]}_{args["model"]}_{args["type"]}_seed{args["seed"]}'
print(args)
random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])


torch.backends.cudnn.benchmark = True

## load a dev dataset
model_name = args['model']
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
batch_size = args['batch_size'] #2 devices

data_path = os.path.join(args['datadir'], args['task'])
data = torch.load(os.path.join(data_path, f'{args["task"]}_{args["model"]}_{args["split"]}_data.pkl'))
#attention_mask = torch.load(os.path.join(data_path, f'{args["task"]}_{args["model"]}_attention_mask.pkl'))
label = torch.load(os.path.join(data_path, f'{args["task"]}_{args["model"]}_{args["split"]}_label.pkl'))

#adataset_dev = torch.utils.data.TensorDataset(data, label) 
#print(f"Num of Data: {len(dataset_dev)}")
#collate_fn = None #dataset.collate_sequences if flag_rnn else None
#iterator_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size, 
#                                           collate_fn=collate_fn, shuffle=False, pin_memory = True)

iterator_dev = zip(data, label) 
#since here we use a list of songs, 
#each song contain several segments of 128, use the first dim as batch to vote

composer2id = torch.load(os.path.join(data_path, 'composer2id_map.pkl'))    
num_labels = len(composer2id)

config = transformers.AutoConfig.from_pretrained(model_name, num_labels = num_labels)
model = transformers.AutoModelForSequenceClassification.from_config(config)#.to(device)
model.load_state_dict(torch.load(args["state_dict"]))
model.cuda()
#model = torch.nn.DataParallel(model)
if args['shift_table'] != '':
    shift_table = torch.load(args['shift_table']).cuda()

writer = tensorboardX.SummaryWriter(log_dir=args['logdir'], 
                                    filename_suffix=f'_{args["split"]}_{args["task"]}_{args["type"]}_seed{args["seed"]}')
model = model.eval()

with torch.no_grad():
    dev_loss = 0
    dev_acc = 0
    for b, (input_ids, labels) in enumerate(tqdm(iterator_dev, total = label.shape[0])):
        input_ids = input_ids.to(device)
        if args['shift_table']!= '':
            input_ids = shift_table(input_ids).long().squeeze()
        #attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids = input_ids, 
                        return_dict = True)
        logits = outputs.logits
        #loss = loss.mean()*input_ids.shape[0]
        #dev_loss += loss.item()
        ans = torch.mode(torch.argmax(logits[0], dim = -1))
        ans = ans.values
        dev_acc = dev_acc + torch.sum(torch.eq(ans, labels)).item()
    #print(f'loss: {dev_loss/len(dataset_dev)}; acc:{dev_acc/len(dataset_dev)}')
    print(f'acc:{dev_acc/label.shape[0]}')
    #writer.add_scalar(f'{args["split"]}_loss', dev_loss/len(dataset_dev), args['step'])
    writer.add_scalar(f'{args["split"]}_acc', dev_acc/label.shape[0], args['step'])
    writer.close()
dev_result = dev_acc/label.shape[0]
output_data = pd.DataFrame([[dev_result]], columns=[f'{args["split"]}_result'])
state_dict_dir = os.path.dirname(args["state_dict"])
state_dict_name = os.path.basename(args["state_dict"])
data_name = state_dict_name.replace(".pkl", f'_{args["split"]}.csv')
output_data.to_csv(os.path.join(state_dict_dir, data_name))

