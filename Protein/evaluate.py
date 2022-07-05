import src.config as config
from src.data.alphabets import Protein
import src.data.localization as localization
import src.data.fluorescence as fluorescence
import src.data.solubility as solubility
import src.data.secstr as secstr
import src.data.stability as ss
import src.data.transmembrane as transmembrane
import src.data.mydataset as mydataset
from src.utils import Print, set_seeds, set_output, load_models
from finetune import valid

import os
import scipy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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
parser.add_argument('--step', type = int, default = -1, help = 'load the checkpoints of different fine-tuning steps')

parser.add_argument('--logdir', type = str, default = './log')
args = vars(parser.parse_args())

#if args['filename'] == None:
#    args['filename'] = f'{args["task"]}_{args["model"]}_{args["type"]}_seed{args["seed"]}'
print(args)

args["data_config"] = f'./config/data/{args["task"]}.json'
args["sanity_check"] = False

set_seeds(args['seed'])
torch.backends.cudnn.benchmark = True

LOAD_FUNCTION_MAP = {
    "localization": localization.load_localization,
    "transmembrane": transmembrane.load_transmembrane,
    "secstr": secstr.load_secstr,
    "solubility": solubility.load_solubility,
    "stability": ss.load_stability,
    "fluorescence": fluorescence.load_fluorescence
}
MAX_LEN = {
    "localization":512,
    "fluorescence":256,
    "stability":50
}
load = LOAD_FUNCTION_MAP[args['task']]

alphabet = Protein()
cfgs = []
data_cfg  = config.DataConfig(args["data_config"])
cfgs.append(data_cfg)

## load a dev dataset
model_name = args['model']
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
batch_size = args['batch_size'] #2 devices

dataset_dev = load(data_cfg, args['split'], alphabet, False)
dataset_dev = mydataset.Seq_dataset(*dataset_dev, encoder = alphabet, tokenizer = tokenizer, 
                                      args = args, max_len=MAX_LEN[args['task']], cache_dir = f'./preprocess_input/{args["task"]}',
                                      split = args['split'])
collate_fn = None #dataset.collate_sequences if flag_rnn else None
iterator_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, pin_memory = True)

config = transformers.AutoConfig.from_pretrained(model_name, num_labels = data_cfg.num_labels)
model = transformers.AutoModelForSequenceClassification.from_config(config)#.to(device)
model.load_state_dict(torch.load(args["state_dict"]))
model.cuda()
#model = torch.nn.DataParallel(model)
if args['shift_table'] != '':
    shift_table = torch.load(args['shift_table']).cuda()

writer = tensorboardX.SummaryWriter(log_dir=args['logdir'], 
                                    filename_suffix=f'_{args["split"]}_{args["task"]}_{args["type"]}_seed{args["seed"]}_bert-large-uncased')

dev_result = valid(model = model,
                   data_cfg = data_cfg,
                   dataloader = iterator_dev,
                   shift_table = shift_table,
                   writer = writer,
                   step = args['step'],
                   args = args)

output_data = pd.DataFrame([[dev_result]], columns=[f'{args["split"]}_result'])
state_dict_dir = os.path.dirname(args["state_dict"])
state_dict_name = os.path.basename(args["state_dict"])
data_name = state_dict_name.replace(".pkl", f'_{args["split"]}.csv')
output_data.to_csv(os.path.join(state_dict_dir, data_name))
#model = model.eval()
#
#with torch.no_grad():
#    if data_cfg.num_labels > 1:
#        dev_loss = 0
#        dev_acc = 0
#        for b, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(iterator_dev)):
#            input_ids = input_ids.to(device)
#            if args['shift_table']!= '':
#                input_ids = shift_table(input_ids).long().squeeze()
#            token_type_ids = token_type_ids.to(device)
#            attention_mask = attention_mask.to(device)
#            labels = labels.to(device)
#
#            loss, logits = model(input_ids = input_ids, 
#                                 token_type_ids = token_type_ids, 
#                                 attention_mask = attention_mask,
#                                 labels = labels)
#            loss = loss.mean()*input_ids.shape[0]
#            dev_loss += loss.item()
#            ans = torch.argmax(logits, dim = -1, keepdim = True)
#            dev_acc = dev_acc + torch.sum(torch.eq(ans, labels)).item()
#        print(f'loss: {dev_loss/len(dataset_dev)}; acc:{dev_acc/len(dataset_dev)}')
#        writer.add_scalar(f'{args["split"]}_loss', dev_loss/len(dataset_dev), args['step'])
#        writer.add_scalar(f'{args["split"]}_acc', dev_acc/len(dataset_dev), args['step'])
#        writer.close()
#
#    else:
#        prediction = []
#        ground = []
#        dev_loss = 0
#        for b, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(iterator_dev)):
#            input_ids = input_ids.to(device)
#            if args['shift_table']!='':
#                input_ids = shift_table(input_ids).long().squeeze()
#            token_type_ids = token_type_ids.to(device)
#            attention_mask = attention_mask.to(device)
#            labels = labels.to(device).view(-1)
#            loss, logits = model(input_ids = input_ids, 
#                                 token_type_ids = token_type_ids, 
#                                 attention_mask = attention_mask,
#                                 labels = labels)
#            loss = loss.mean()*input_ids.shape[0]
#            dev_loss += loss.item()
#            prediction.append(logits.view(-1).cpu())
#            ground.append(labels.cpu())
#        prediction = torch.cat(prediction).numpy()
#        ground = torch.cat(ground).numpy()
#        spearman_r = scipy.stats.spearmanr(prediction, ground)
#        print(spearman_r)
#        writer.add_scalar(f'{args["split"]}_loss', dev_loss/len(dataset_dev), args['step'])
#        writer.add_scalar(f'{args["split"]}_spearman_r', spearman_r[0], args['step'])
#        writer.close()
