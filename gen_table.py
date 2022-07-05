import torch
import transformers
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str)
parser.add_argument('--seed', type = int)
parser.add_argument('--save_dir', type = str, default='./assign_token/new_random/')
args = parser.parse_args()
torch.manual_seed(args.seed)
name = f'table_seed{args.seed}.pkl'
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
token_ids = list(range(tokenizer.vocab_size))
unused = token_ids[:1000]
token_ids = token_ids[1000:]
random.shuffle(token_ids)
token_ids.extend(unused)
token_ids = torch.Tensor(token_ids).unsqueeze(1)
table = torch.nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=1)
table.weight = torch.nn.Parameter(token_ids)
torch.save(table, f'{os.path.join(args.save_dir, name)}')
