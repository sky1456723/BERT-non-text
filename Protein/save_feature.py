import src.config as config
from src.data.alphabets import Protein
import src.data.localization as localization
import src.data.fluorescence as fluorescence
import src.data.solubility as solubility
import src.data.secstr as secstr
import src.data.stability as ss
import src.data.transmembrane as transmembrane

import src.data.mydataset as mydataset
import src.model.plus_rnn as plus_rnn
import src.model.plus_tfm as plus_tfm
import src.model.p_elmo as p_elmo
import src.model.mlp as mlp
from src.utils import Print, set_seeds, set_output, load_models
from src.train import Trainer
import os
import torch
import argparse
import tensorboardX
#import torch.utils.tensorboard
#from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers.optimization import get_linear_schedule_with_warmup 
from tqdm import tqdm
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str)
parser.add_argument('--model', type = str, default = 'bert-base-uncased')
parser.add_argument('--ratio', type = float, default = 1.0)
#### These parameters are not used by save_feature.py ####
parser.add_argument('--type', type = str, choices=['pretrain', 'scratch'])
parser.add_argument('--seed', type = int, default = 2020)
parser.add_argument('--gradient_accumulation', '-a', type = int, default = 2)
parser.add_argument('--batch_size', '-b', type = int, default = 1)
parser.add_argument('--warmup_step', type = int, default = 0)
parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--logdir', type = str, default = './log')
##########################################################

parser.add_argument('--savedir', type = str, default = './preprocess_input')
#parser.add_argument('--filename', type = str)

parser.add_argument('--split', type = str, choices=['train', 'dev', 'test'])
args = vars(parser.parse_args())
args["savedir"] = os.path.join(args["savedir"], args["task"])
#if args['filename'] == None:
#    args['filename'] = f'{args["task"]}_{args["model"]}_{args["type"]}_seed{args["seed"]}.pkl'
print(args)

LOAD_FUNCTION_MAP = {
    "localization": localization.load_localization,
    "transmembrane": transmembrane.load_transmembrane,
    "secstr": secstr.load_secstr,
    "solubility": solubility.load_solubility,
    "stability": ss.load_stability,
    "fluorescence": fluorescence.load_fluorescence
}
MAX_LEN = {
    "localization": 512,
    "fluorescence": 256,
    "stability": 50
}
load = LOAD_FUNCTION_MAP[args['task']]

set_seeds(args['seed'])
args["data_config"] = f'./config/data/{args["task"]}.json'
args["sanity_check"] = False
alphabet = Protein()
cfgs = []
data_cfg  = config.DataConfig(args["data_config"])
cfgs.append(data_cfg)
#output, save_prefix = set_output(args, "train_localization_log")
batch_size = 1

## load a train dataset
model_name = args['model']
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
if type(args['split']) == type(None):
    splits = ['train', 'dev', 'test']
else:
    splits = [args['split']]
for split in splits:
    dataset_train = load(data_cfg, split, alphabet, False)
    dataset_train = mydataset.Seq_dataset(*dataset_train, encoder = alphabet, tokenizer = tokenizer, 
                                          args = args, max_len=MAX_LEN[args["task"]], cache_dir = f'./preprocess_input/{args["task"]}',
                                          split = split)
    collate_fn = None #dataset.collate_sequences if flag_rnn else None
    iterator_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    for b, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(iterator_train)):
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
    
    input_ids_list = torch.cat(input_ids_list, dim = 0)
    token_type_ids_list = torch.cat(token_type_ids_list, dim = 0)
    attention_mask_list = torch.cat(attention_mask_list, dim = 0)
    total_len = int(input_ids_list.shape[0]*args["ratio"])
    input_ids_list = input_ids_list[:total_len,:]
    token_type_ids_list = token_type_ids_list[:total_len,:]
    attention_mask_list = attention_mask_list[:total_len,:]

    torch.save(input_ids_list, os.path.join(args["savedir"], f'cached_{split}_input_ids_{args["task"]}_{args["model"]}_{MAX_LEN[args["task"]]}.pkl'))
    torch.save(token_type_ids_list, os.path.join(args["savedir"], f'cached_{split}_token_type_{args["task"]}_{args["model"]}_{MAX_LEN[args["task"]]}.pkl'))
    torch.save(attention_mask_list, os.path.join(args["savedir"], f'cached_{split}_att_mask_{args["task"]}_{args["model"]}_{MAX_LEN[args["task"]]}.pkl'))


