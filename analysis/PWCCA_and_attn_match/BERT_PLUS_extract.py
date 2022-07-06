import sys
import os
sys.path.append('./PLUS_config/')
sys.path.append('./PLUS_config/src')
import argparse
import transformers
import torch
import copy
import json
import scipy
import numpy as np

from tqdm import tqdm

import src_plus.config as dataconfig
from src_plus.data.alphabets import Protein
import src_plus.data.localization as localization
import src_plus.data.fluorescence as fluorescence
import src_plus.data.stability as ss
import src_plus.data.mydataset as mydataset
#import src_plus.data.dataset as dataset
from src_plus.train import Trainer
from src_plus.utils import Print, set_seeds, set_output, load_models
import src_plus.model.plus_tfm as plus_tfm
import src_plus.model.mlp as mlp
from src_plus.utils import Print, set_seeds, set_output, load_models
from utils import PWCCA

torch.manual_seed(100)

parser = argparse.ArgumentParser()
parser.add_argument('--protein_dir', type = str)
parser.add_argument('--save_dir', type = str)
parser.add_argument('--pretrained_BERT', action = 'store_true')
parser.add_argument('--bert_ckpt', type = str, default = '')
parser.add_argument('--scratch_ckpt', type = str, default = '')
parser.add_argument('--plus_ckpt', type = str, default = '')
parser.add_argument('--task', type = str)
parser.add_argument('--feature', type = str, choices=['hidden', 'attention'])
input_args = parser.parse_args()

args = {}
args["model"] = "bert-base-uncased"
args["task"] = input_args.task
args["data_config"] = f'./PLUS_config/data/{args["task"]}.json'
args["model_config"] = f'./PLUS_config/model/plus-tfm.json'
args["run_config"] = f'./PLUS_config/run/plus-tfm_others.json'
args["sanity_check"] = True
args["pretrained_model"] = input_args.plus_ckpt #f'./pretrained_plus/PLUS-TFM.pt'
LOAD_FUNCTION_MAP = {
        "localization": localization.load_localization,
        "stability": ss.load_stability,
        "fluorescence": fluorescence.load_fluorescence
}
load = LOAD_FUNCTION_MAP[args['task']]

d_conf = json.load(open(args["data_config"]))
config = transformers.BertConfig.from_pretrained('bert-base-uncased', 
                                                 num_labels = d_conf['num_labels'])
if input_args.pretrained_BERT:
    bert = transformers.BertForSequenceClassification.from_pretrained(args["model"])
else:
    bert = transformers.BertForSequenceClassification(config=config)
scratch = transformers.BertForSequenceClassification(config=config)

if input_args.bert_ckpt != '':
    bert.load_state_dict(torch.load(input_args.bert_ckpt))
if input_args.scratch_ckpt != '':
    scratch.load_state_dict(torch.load(input_args.scratch_ckpt))

alphabet = Protein()
cfgs = []
data_cfg  = dataconfig.DataConfig(args["data_config"])
data_cfg.path['dev'] = os.path.join(input_args.protein_dir, data_cfg.path['dev'])
data_cfg.path['train'] = os.path.join(input_args.protein_dir, data_cfg.path['train'])
data_cfg.path['test'] = os.path.join(input_args.protein_dir, data_cfg.path['test'])
cfgs.append(data_cfg)

model_cfg = dataconfig.ModelConfig(args["model_config"], input_dim=len(alphabet), num_classes=1)
cfgs.append(model_cfg)

run_cfg = dataconfig.RunConfig(args["run_config"], sanity_check=args["sanity_check"]);  cfgs.append(run_cfg)
run_cfg.batch_size_eval = 25

if input_args.plus_ckpt != '':
    plus = plus_tfm.PLUS_TFM(model_cfg)
    models_list = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]
    flag_lm_model = False
    flag_rnn = False
    models_list.append([plus, "", flag_lm_model, flag_rnn, False])
    device = 'cuda:0'
    data_parallel = False
    output = sys.stdout
    load_models(args, models_list, device, data_parallel, output, tfm_cls=flag_rnn)

model_name = args['model']
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
dataset_dev = load(data_cfg, 'test', alphabet, False)
dataset_dev = mydataset.PLUS_Seq_dataset(*dataset_dev, alphabet, run_cfg, flag_rnn, model_cfg.max_len, augment=False)
collate_fn = mydataset.collate_sequences if flag_rnn else None
iterator_dev = torch.utils.data.DataLoader(dataset_dev, run_cfg.batch_size_eval, collate_fn=collate_fn,
                                          shuffle = False)

dataset_dev = load(data_cfg, 'test', alphabet, False)
dataset_dev = mydataset.Seq_dataset(*dataset_dev, 
                                    encoder = alphabet, 
                                    tokenizer = tokenizer,
                                    args = args, 
                                    max_len=512, 
                                    cache_dir = f'{input_args.protein_dir}/preprocess_input/{args["task"]}',
                                    split = 'test')
shift_table = torch.load(f'../../assign_token/{args["model"]}/table_seed100.pkl')
def get_feature(bert, dataset, shift_table, feat='hidden'):
    total_len = 0
    
    collate_fn = None #dataset.collate_sequences if flag_rnn else None
    iterator_dev = torch.utils.data.DataLoader(dataset_dev, 
                                               batch_size=20, 
                                               collate_fn=collate_fn, 
                                               shuffle=False, 
                                               pin_memory = True)
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    bert = bert.cuda(0)
    seq_len_list = []
    if feat == 'hidden':
        bert_hidden = []
    elif feat == 'attention':
        bert_hidden = [ [] for i in range(12)]
    with torch.no_grad():
        bert_loss = []
        scratch_loss = []
        for b, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(iterator_dev)):
            if b == 100: #if b == 100:
                break
            input_ids = shift_table(input_ids).long().squeeze()
            input_ids = input_ids.cuda(0)
            token_type_ids = token_type_ids.cuda(0)
            attention_mask = attention_mask.cuda(0)
            seq_len = torch.sum(attention_mask, dim = -1)
            #labels = labels.cuda(0)

            bert_out = bert(input_ids = input_ids, 
                             token_type_ids = token_type_ids, 
                             attention_mask = attention_mask,
                             output_hidden_states = True,
                             output_attentions = True,
                             return_dict=True)
            seq_len = torch.sum(attention_mask, dim = -1)
            seq_len_list.append(seq_len.cpu())
            if feat == 'hidden':
                for i in range(bert_out.hidden_states[-1].shape[0]):
                    bert_hidden.append(bert_out.hidden_states[-1][i,:seq_len[i]].cpu())
                    total_len += seq_len[i]
            elif feat == 'attention':
                for l in range(12):
                    attn = bert_out.attentions[l].cpu()
                    #attn = attn.view(-1, attn.shape[2], attn.shape[3])
                    bert_hidden[l].append(attn)
                    
                bsz = bert_out.attentions[0].shape[0]
                #n_head = bert_out.attentions[0].shape[1]
                total_len += bsz #* n_head
                print(total_len)
            if total_len >= 100:
                if feat == 'hidden':
                    bert_hidden = torch.cat(bert_hidden)[:1000,:]
                elif feat == 'attention':
                    for l in range(12):
                        bert_hidden[l] = torch.cat(bert_hidden[l], dim = 0)[:100]
                break
    bert = bert.cpu()
    with torch.cuda.device('cuda:0'):
        torch.cuda.empty_cache()
    return bert_hidden, seq_len_list

bert_hidden, bert_seq_len = get_feature(bert = bert, 
                          dataset = dataset_dev, 
                          shift_table=shift_table,
                          feat = input_args.feature)

scratch_hidden, _ = get_feature(bert = scratch, 
                             dataset = dataset_dev, 
                             shift_table=shift_table,
                             feat = input_args.feature)


if input_args.plus_ckpt != '':
    total_len = 0
    atten_scores = {i:[] for i in range(12)}
    plus_seq_len = []
    with torch.no_grad():
        plus_embeddings = []
        for b, batch in enumerate(tqdm(iterator_dev)):
            if b == 100: #if b == 80:
                break
            batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
            if len(batch) == 4:
                tokens, segments, input_mask, labels = batch
                masked_pos,  per_seq = None, True
            elif len(batch) == 6:
                tokens, segments, input_mask, labels, valids, label_weights = batch
                masked_pos, per_seq = None, False
            elif len(batch) == 7:
                tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights, labels = batch
                per_seq = True
            elif len(batch) == 9:
                tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights, labels, valids, label_weights = batch
                per_seq = False
            #logits_lm, logits_cls = plus(tokens, segments, input_mask, masked_pos, per_seq)
            h = plus(tokens, segments, input_mask, masked_pos, per_seq, embedding = True)
            seq_len = torch.sum(input_mask, dim = -1)
            for i in range(h.shape[0]):
                plus_embeddings.append(h[i,:seq_len[i]].cpu())
                #total_len += seq_len[i].item()
            for l in range(12):
                scores = plus.transformer.blocks[l].attn.scores.cpu()
                #scores = scores.view(-1, scores.shape[2], scores.shape[3])
                atten_scores[l].append(scores)
            plus_seq_len.append(seq_len)
            bsz = plus.transformer.blocks[0].attn.scores.shape[0]
            n_head = plus.transformer.blocks[0].attn.scores.shape[1]
            total_len += bsz#*n_head
            if total_len >= 100:
                plus_embeddings = torch.cat(plus_embeddings)[:1000,:]
                for l in range(12):
                    atten_scores[l] = torch.cat(atten_scores[l], dim = 0)[:100]
                break
    plus = plus.cpu()
    with torch.cuda.device('cuda:0'):
        torch.cuda.empty_cache()


os.makedirs(input_args.save_dir, exist_ok = True)
torch.save(bert_hidden, os.path.join(input_args.save_dir, "bert_hidden.pkl"))
torch.save(scratch_hidden, os.path.join(input_args.save_dir, "scratch_hidden.pkl"))
if input_args.plus_ckpt != '':
    torch.save(plus_embeddings, os.path.join(input_args.save_dir, "plus_hidden.pkl"))
    torch.save(atten_scores, os.path.join(input_args.save_dir, "plus_attn.pkl"))