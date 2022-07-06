import torch
from utils import PWCCA

task = 'localization'

bert_hidden = torch.load('./features/bert_hidden.pkl')
scratch_hidden = torch.load('./features/scratch_hidden.pkl')
plus_embeddings = torch.load('./attn_features/plus_hidden.pkl')

regs={'stability':1.5e-3, 'fluorescence':1e-3, 'localization':1e-4}
results = PWCCA(X = bert_hidden, Y = plus_embeddings, reg = regs[task])
#results2 = PWCCA(X = bert2_hidden, Y = plus_embeddings, reg = regs[args["task"]])
#bert2bert = PWCCA(X = bert_hidden, Y = bert2_hidden, reg = regs[args["task"]])
protein2s = PWCCA(X = scratch_hidden, Y = plus_embeddings, reg = regs[task])
#s2s = PWCCA(X = scratch_hidden, Y = scratch2_hidden, reg = regs[args["task"]])
scratch_results = PWCCA(X = bert_hidden, Y = scratch_hidden, reg = regs[task])


print("bert-plus: ", results['weighted_cca'])
print("bert-scratch: ", scratch_results['weighted_cca'])
print("scratch-plus: ", protein2s['weighted_cca'])