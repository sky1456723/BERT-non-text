import os
import scipy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import copy
import argparse
import tensorboardX
#import torch.utils.tensorboard
#from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers.optimization import get_linear_schedule_with_warmup 
from tqdm import tqdm
import time

import src.config as dataconfig
from src.data.alphabets import Protein
import src.data.localization as localization
import src.data.fluorescence as fluorescence
import src.data.stability as ss
import src.data.mydataset as mydataset
from src.utils import Print, set_seeds, set_output, load_models

def main():
#device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = str, help = "Protrein classification task name")
    parser.add_argument('--model', type = str, default = 'bert-base-uncased', help = "pre-trained model to load")
    parser.add_argument('--type', type = str, choices=['pretrain', 'scratch'], help = "load pre-trained model or not")
    parser.add_argument('--seed', type = int, default = 2020, help = "random seed")
    parser.add_argument('--gradient_accumulation', '-a', type = int, default = 2)
    parser.add_argument('--batch_size', '-b', type = int, default = 16)
    parser.add_argument('--epoch', '-e', type = int, default = 20)
    parser.add_argument('--total_step', type = int, default = 1e10)
    parser.add_argument('--warmup_step', type = int, default = 0)
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'learning rate')
    #parser.add_argument('--shift', type = int, default = 0, help = 'the constant c for the "shift c" setting')
    parser.add_argument('--shift_table', type = str, default = '', help = 'the table file for the token mapping')
    parser.add_argument('--rand_embed', action = 'store_true', help = 'run the experiment for randomized embedding')
    parser.add_argument('--n_gpu', type = int)
    parser.add_argument('--ckpt', type = str, default = '', help = 'ckpt file for the experiment of different pre-training steps')
    
    parser.add_argument('--logdir', type = str, default = './log')
    parser.add_argument('--savedir', type = str, default = './save_model')
    parser.add_argument('--save_step', type = int, default = 3000)
    parser.add_argument('--filename', type = str)
    parser.add_argument('--postfix', type = str, default = '')
    args = vars(parser.parse_args())
    os.makedirs(args['savedir'], exist_ok = True)
    torch.save(args, f'{os.path.join(args["savedir"], "args.pkl")}')
    if args['filename'] == None:
        args['filename'] = f'{args["task"]}_{args["model"]}_{args["type"]}_seed{args["seed"]}'
    #if args['shift']!=0:
    #    args['filename'] += f'_shift{args["shift"]}'
    if args['shift_table']!='':
        args['filename'] += '_table_'
    args['filename'] += args['postfix']
    print(args)

    args["data_config"] = f'./config/data/{args["task"]}.json'
    args["sanity_check"] = False
    
    set_seeds(args['seed'])
    torch.backends.cudnn.benchmark = True

    LOAD_FUNCTION_MAP = {
        "localization": localization.load_localization,
        "stability": ss.load_stability,
        "fluorescence": fluorescence.load_fluorescence
    }
    MAX_LEN = {
        "localization": 512,
        "stability": 50,
        "fluorescence": 256
    }
    load = LOAD_FUNCTION_MAP[args['task']]

    alphabet = Protein()
    cfgs = []
    data_cfg  = dataconfig.DataConfig(args["data_config"])
    cfgs.append(data_cfg)

    ## load a train dataset
    model_name = args['model']
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    batch_size = args['batch_size'] #2 devices
    epoch = args['epoch']
    gradient_accumulation = args['gradient_accumulation']

    dataset_train = load(data_cfg, "train", alphabet, False)
    dataset_train = mydataset.Seq_dataset(*dataset_train, encoder = alphabet, tokenizer = tokenizer,
                                         args = args, max_len=MAX_LEN[args['task']], cache_dir = f'./preprocess_input/{args["task"]}',
                                          split = 'train')
    collate_fn = None #dataset.collate_sequences if flag_rnn else None
    iterator_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, pin_memory = True, drop_last = True)

    dataset_dev = load(data_cfg, 'dev', alphabet, False)
    dataset_dev = mydataset.Seq_dataset(*dataset_dev, encoder = alphabet, tokenizer = tokenizer,
                                      args = args, max_len=MAX_LEN[args['task']], cache_dir = f'./preprocess_input/{args["task"]}',
                                      split = 'dev')
    collate_fn = None #dataset.collate_sequences if flag_rnn else None
    iterator_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, pin_memory = True)
    config = transformers.AutoConfig.from_pretrained(model_name, num_labels = data_cfg.num_labels)
    if args['type'] == 'pretrain':
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                                num_labels = data_cfg.num_labels)#.to(device)
    else:
        model = transformers.AutoModelForSequenceClassification.from_config(config)#.to(device)

    if args['ckpt'] != '':
        state_dict = torch.load(args['ckpt'])
        pretrain_config = transformers.AutoConfig.from_pretrained(model_name)
        pretrain_model = transformers.AutoModelForPreTraining.from_pretrained(None,
                                                                           state_dict = state_dict,
                                                                           config = pretrain_config)
        try:
            model.bert = copy.deepcopy(pretrain_model.bert)
        except:
            model.albert = copy.deepcopy(pretrain_model.albert)
        del pretrain_model, state_dict
        print(f"[finetune] Pretrain checkpoint loaded from {args['ckpt']}")
    if args['rand_embed']:
        scratch_model = transformers.AutoModelForSequenceClassification.from_config(config)
        model.bert.embeddings.word_embeddings = copy.deepcopy(scratch_model.bert.embeddings.word_embeddings)
        print(f"[finetune] Word embedding randomized.")
    

    writer = tensorboardX.SummaryWriter(log_dir=args['logdir'],
                                        filename_suffix=f'_train_{args["task"]}_{args["type"]}_seed{args["seed"]}')
    logging_step = 50
    if args['shift_table']!='':
        shift_table = torch.load(args['shift_table'])
        shift_table.cuda()
        print(f"[finetune.py] Shift table loaded from {args['shift_table']}")
    train(data_cfg = data_cfg, model = model, train_loader = iterator_train, valid_loader = iterator_dev, 
          shift_table = shift_table, writer = writer, logging_step = logging_step, args = args)

def valid(data_cfg, model, dataloader, shift_table, writer, args, step):
    print("Validation ", step)
    model = model.eval()
    with torch.no_grad():
        #if data_cfg.num_labels > 1:
        dev_loss = 0
        prediction = []
        ground = []
        dev_acc = 0
        for b, (input_ids, token_type_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.cuda()
            if args['shift_table']!= '':
                input_ids = shift_table(input_ids).long().squeeze()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
    
            output = model(input_ids = input_ids, 
                           token_type_ids = token_type_ids, 
                           attention_mask = attention_mask,
                           labels = labels,
                           return_dict = True)
            loss = output.loss
            logits = output.logits
            loss = loss.mean()*input_ids.shape[0]
            dev_loss += loss.item()
            if data_cfg.num_labels > 1:
                prediction.append(torch.argmax(logits, dim = -1).cpu())
            else:
                prediction.append(logits.view(-1).cpu())
            ground.append(labels.cpu())
        writer.add_scalar('dev_loss', dev_loss/len(dataloader.dataset), step)
        prediction = torch.cat(prediction)
        ground = torch.cat(ground).view(-1)
        if data_cfg.num_labels > 1:
            dev_acc = torch.mean(torch.eq(prediction, ground).float()).item()
            print(f'loss: {dev_loss/len(dataloader.dataset)}; acc:{dev_acc}')
            writer.add_scalar('dev_acc', dev_acc, step)
            dev_result = dev_acc
        else:
            prediction = prediction.numpy()
            ground = ground.numpy()
            spearman_r = scipy.stats.spearmanr(prediction, ground)
            print(f'loss: {dev_loss/len(dataloader.dataset)}; spearman:{spearman_r[0]}')
            writer.add_scalar('dev_spearman_r', spearman_r[0], step)
            dev_result = spearman_r[0]
        model.train()
        return dev_result


def train(data_cfg, model, train_loader, valid_loader, shift_table, writer, logging_step, args):
    model.cuda()
    if args['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    epoch = args['epoch']
    gradient_accumulation = args['gradient_accumulation']
    optimizer = torch.optim.Adam(model.parameters(), lr = args['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, args['warmup_step'], len(train_loader)*epoch/gradient_accumulation)
    model.train()
    print("Model.train(): ", model.training)
    global_step = 0
    update_step = 0
    last_step = 0
    #batch_acc = 0
    logging_loss = 0
    tr_loss = 0
    dev_best = 0
    optimizer.zero_grad()
    for e in range(epoch):
        print("Epoch: ", e)
        print("")
        for b, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(train_loader)):
            input_ids = input_ids.cuda(non_blocking=False)
            with torch.no_grad():
                if args['shift_table']!='':
                    input_ids = shift_table(input_ids).to(torch.long).squeeze()
                elif args['shift']!=0:
                    raise NotImplementedError
                    #input_ids = torch.remainder(input_ids + args['shift'], model.module.config.vocab_size)
            token_type_ids = token_type_ids.cuda(non_blocking=False)
            attention_mask = attention_mask.cuda(non_blocking=False)
            labels = labels.cuda(non_blocking=False)
            output = model(input_ids = input_ids, 
                           token_type_ids = token_type_ids, 
                           attention_mask = attention_mask,
                           labels = labels,
                           return_dict = True)
            loss = output.loss
            logits = output.logits
            if args['n_gpu'] > 1:
                loss = loss.mean()
            loss = loss/gradient_accumulation
            loss.backward()
            
            tr_loss += loss.item()
            global_step += 1
            if global_step % gradient_accumulation == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                update_step += 1
                if update_step % logging_step == 0:
                    writer.add_scalar('loss', (tr_loss - logging_loss)/logging_step, update_step)

                    #print(f"[step {update_step}] loss: {};\batch acc: {batch_acc.item()/batch_size}")
                    logging_loss = tr_loss
                if update_step % args['save_step'] == 0:
                    dev_result = valid(model = model, 
                                       data_cfg = data_cfg,
                                       dataloader = valid_loader, 
                                       shift_table = shift_table,
                                       writer = writer,
                                       step = update_step,
                                       args = args)
                    if args['n_gpu'] > 1:
                        torch.save(model.module.state_dict(), os.path.join(args['savedir'], args['filename']+'_'+str(update_step)+'.pkl'))
                        if dev_result > dev_best:
                            dev_best = dev_result
                            print("[finetune.py] Saving dev_best.pkl")
                            torch.save(model.module.state_dict(), os.path.join(args['savedir'], f'{args["type"]}_dev_best.pkl'))
                    else:
                        torch.save(model.state_dict(), os.path.join(args['savedir'], args['filename']+'_'+str(update_step)+'.pkl'))
                        if dev_result > dev_best:
                            print("[finetune.py] Saving dev_best.pkl")
                            dev_best = dev_result
                            torch.save(model.state_dict(), os.path.join(args['savedir'], f'{args["type"]}_dev_best.pkl'))
                if update_step >= args['total_step']:
                    if args['n_gpu'] > 1:
                        torch.save(model.module.state_dict(), os.path.join(args['savedir'], args['filename']+'_'+str(update_step)+'.pkl'))
                    else:
                        torch.save(model.state_dict(), os.path.join(args['savedir'], args['filename']+'_'+str(update_step)+'.pkl'))
                    return 0
    dev_result = valid(model = model, 
                       data_cfg = data_cfg,
                       dataloader = valid_loader, 
                       shift_table = shift_table,
                       writer = writer,
                       step = update_step,
                       args = args)

    if args['n_gpu'] > 1:
        torch.save(model.module.state_dict(), os.path.join(args['savedir'], args['filename']+'_'+str(update_step)+'.pkl'))
        if dev_result > dev_best:
            dev_best = dev_result
            print("[finetune.py] Saving dev_best.pkl")
            torch.save(model.module.state_dict(), os.path.join(args['savedir'], f'{args["type"]}_dev_best.pkl'))
    else:
        torch.save(model.state_dict(), os.path.join(args['savedir'], args['filename']+'_'+str(update_step)+'.pkl'))
        if dev_result > dev_best:
            print("[finetune.py] Saving dev_best.pkl")
            dev_best = dev_result
            torch.save(model.state_dict(), os.path.join(args['savedir'], f'{args["type"]}_dev_best.pkl'))

if __name__ == '__main__':
    main()
