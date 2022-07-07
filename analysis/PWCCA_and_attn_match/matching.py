import torch
import scipy
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

@torch.no_grad()
def costing(x, y):
    n_layer = len(x)
    n_data = x[0].shape[0]
    n_head = x[0].shape[1]
    all_cost = []
    for i in range(n_layer):
        p = x[i].reshape(n_data, n_head, -1).unsqueeze(1).cuda(0)
        q = y[i].reshape(n_data, n_head, -1).unsqueeze(2).cuda(0)
        #print(p.shape, q.shape)
        cost = []
        for n in range(n_data):
            cost.append(torch.sum(torch.abs(p[n]-q[n]), dim = -1).cpu())
        all_cost.append(torch.stack(cost, dim = 0).numpy())
        #cost = torch.sum((p-q)**2, dim = -1)
        #all_cost.append(cost)
    return all_cost


def matching(c):
    match = []
    total_cost = []
    for l in range(len(c)):
        match.append([])
        total_cost.append([])
        for i in range(c[0].shape[0]):
            row, col = scipy.optimize.linear_sum_assignment(c[l][i])
            match[l].append((row, col))
            total_cost[l].append(c[l][row, col].sum())
    total_cost = np.array(total_cost)
    mean_cost = np.mean(total_cost, axis = -1)
    std_cost = np.std(total_cost, axis = -1)
    return mean_cost, std_cost, match

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str)
args = parser.parse_args()

bert_attn = torch.load(os.path.join(args.data_dir, 'bert_hidden.pkl'))
scratch_attn = torch.load(os.path.join(args.data_dir, 'scratch_hidden.pkl'))
plus_attn = torch.load(os.path.join(args.data_dir, 'plus_attn.pkl'))

bert_plus_cost = costing(bert_attn, plus_attn)
bert_scratch_cost = costing(bert_attn, scratch_attn)

bert_plus_mean, v1, m1 = matching(bert_plus_cost)
bert_scratch_mean, v2, m2 = matching(bert_scratch_cost)

fig, ax = plt.subplots(figsize=[4.8,4.8])
cmaps = [plt.get_cmap('Blues'), plt.get_cmap('Oranges')]
x = range(len(bert_plus_mean))
ax.fill_between(x, 
                bert_plus_mean - v1, 
                bert_plus_mean + v1, color=cmaps[0](X=0.3, alpha=0.5))
ax.plot(x, bert_plus_mean, '-o', label='BERT-PLUS')
#ax2=ax.twinx()
#ax2.plot(dev_score['Unnamed: 0'], dev_score['mean'], 'o--', label='pretrain')
#ax2.axhline(dev_score['mean'].iloc[-1], linestyle = '--', label='pre-trained')
#ax2.fill_between(dev_score['Unnamed: 0'], 
#                dev_score['mean'] - dev_score['std'], 
#                dev_score['mean'] + dev_score['std'],  color=cmaps[0](X=0.3, alpha=0.5))
#ax2.axhspan(dev_score['mean'].iloc[-1] - dev_score['std'].iloc[-1],
#            dev_score['mean'].iloc[-1] + dev_score['std'].iloc[-1], color=cmaps[0](X=0.3, alpha=0.5))
#ax2.set_ylim(ymin=0, ymax=1)

ax.fill_between(x, 
                bert_scratch_mean - v2, 
                bert_scratch_mean + v2, color=cmaps[1](X=0.3, alpha=0.5))
#ax.plot(x, bert_scratch_mean, '-o', label='BERT-隨機初始化')
ax.plot(x, bert_scratch_mean, '-o', label='BERT-scratch')
#ax2.plot(dev_scratch_score['Unnamed: 0'], dev_scratch_score['mean'], 'o--', label='scratch')
#ax2.fill_between(dev_scratch_score['Unnamed: 0'], 
#                dev_scratch_score['mean'] - dev_scratch_score['std'], 
#                dev_scratch_score['mean'] + dev_scratch_score['std'], color=cmaps[1](X=0.3, alpha=0.5))
#ax2.axhline(dev_scratch_score['mean'].iloc[-1], linestyle = '--', label='scratch', color='orange')
#ax2.axhspan(dev_scratch_score['mean'].iloc[-1] - dev_scratch_score['std'].iloc[-1],
#            dev_scratch_score['mean'].iloc[-1] + dev_scratch_score['std'].iloc[-1], color=cmaps[1](X=0.3, alpha=0.5))
#ax2.set_ylim(ymin=0, ymax=1)
fontsize = 20
#ax.legend(prop=prop)
ax.legend(fontsize=20)
ax.set_ylim(ymin=0)
#ax.set_ylabel('L1距離', fontproperties=prop)
ax.set_ylabel('L1 distance', fontsize=20)
#ax.set_xlabel('模型層', fontproperties=prop)
ax.set_xlabel('layer', fontsize=20)
#if task=='fluorescence' or task=='stability':
#    ax2.set_ylabel('dev spearman r', fontsize = fontsize)
#else:
#    ax2.set_ylabel('dev acc', fontsize = fontsize)
x_tick = [str(i+1) for i in x]
ax.tick_params(axis = 'x', labelsize=fontsize-4)
ax.set_xticks(x)
ax.set_xticklabels(x_tick)
ax.tick_params(axis = 'y', labelsize=fontsize-4)
ax.ticklabel_format(axis= 'y', style='sci', scilimits=(0,0))
ax.yaxis.offsetText.set_fontsize(fontsize-4)
#ax2.tick_params(axis = 'x', labelsize=fontsize-4)
#ax2.tick_params(axis = 'y', labelsize=fontsize-4)
fig.tight_layout()
fig.savefig(f'./pretrain_bert_plus_attention.png')
#ax2.legend(loc=5)
