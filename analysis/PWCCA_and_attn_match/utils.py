import sys
import torch
import numpy as np
import sklearn
import rcca
try:
    import matplotlib
    import matplotlib.pyplot as plt
except:
    print("No matplotlib imported")
from tqdm import tqdm 
import logging
import warnings
from cycler import cycler
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

def print_result(data_dict, dataset = None, word=False, layer_err=False, mask_err = False,
                 next_cls_err=False, file = sys.stdout, mask_num = None, return_list = False):
    
    
    if file!=sys.stdout:
        file = open(file, "w")
    
    if layer_err or mask_err:
        print("### Token acc ###", file = file)
        #length = len(data_dict['layer_1'])
        acc_list  = []
        for k in data_dict.keys():
            #if k=='layer_0':
            #    continue
            if layer_err:
                acc = np.mean(data_dict[k])
            elif mask_err:
                acc = sum(data_dict[k])/mask_num
            acc_list.append(acc)
            print(k,": %.3f" % ( acc ), file = file)
        if return_list:
            return acc_list
    if next_cls_err:
        print("### Token acc (next to CLS)", file = file)
        for k in cls_err.keys():
            print(k,": %.3f" % (data_dict[k]/length), file = file)

    #print()
    if word:
        print("### Oracle ###", file = file)
        print(dataset.tokenizer.decode(dataset[id][0][1:-1]), file = file)
        print( " " , file = file)
        
        id = np.random.randint(0, len(dataset))
        print("### Sample ###", file = file)
        for k in data_dict.keys():
            #if k=='layer_0':
            #    continue
            print(k,": ", data_dict[k][id], file = file)
            print( " " , file = file)
            
            
'''
vector corr
https://academic.oup.com/biomet/article-pdf/66/1/41/600457/66-1-41.pdf
'''

def vector_corr(U, V, norm=None):
    #print(torch.norm(V, dim = 1, keepdim=True).shape)
    if type(norm) == type(None):
        U_norm = torch.norm(U, dim = 1, keepdim=True, p= 2)
        V_norm = torch.norm(V, dim = 1, keepdim=True, p= 2)
    A = torch.matmul(torch.transpose(U/U_norm, 0, 1), V/V_norm)
    A = A/(V.shape[0])
    A = A.to(device)
    #A.requires_grad = True
    #A = A.numpy()
    u, s, v = torch.svd(A)
    #print(s)
    if (s<1e-9).any():
        print(torch.min(s))
        print("GG")
    return torch.sum(s)

def distance_correlation(stack_output):
    '''
    expect input shape: (layer or # of pair to calculate, vector_num, vector_dim)
    '''
    sub = {}
    for j in range(stack_output.shape[0]):
        layer = []
        for i in tqdm(range(stack_output.shape[1])):
            layer.append(torch.norm(stack_output[j] - stack_output[j,i],dim=1).cpu())
        sub[j] = layer

    stack_sub = {}
    for k in sub.keys():
        stack_sub[k] = torch.stack(sub[k], dim =1)

    means ={}
    for k in stack_sub.keys():
        means[k] = [torch.mean(stack_sub[k], dim = 0, keepdim=True), torch.mean(stack_sub[k], dim = 1, keepdim=True), torch.mean(stack_sub[k])]
        stack_sub[k] = stack_sub[k] - means[k][0]- means[k][1]+ means[k][2]

    cov_xy = []
    for i in tqdm(range(stack_output.shape[0])):
        cov_xy.append([])
        for j in range(stack_output.shape[0]):
            cov_xy[-1].append( torch.mean(stack_sub[i]*stack_sub[j]).item())

    dist_corr = []
    for i in tqdm(range(stack_output.shape[0])):
        dist_corr.append([])
        for j in range(stack_output.shape[0]):
            dist_corr[-1].append( (cov_xy[i][j]/(np.sqrt(cov_xy[i][i]*cov_xy[j][j]))))

    dist_corr = np.array(dist_corr)
    return dist_corr

def do_CCA(X, Y, n_component, reg=0.0, kernel=False, ktype=None, verbose = True):
    # (number of samples X number of features)
    # need to be zero mean
    cca = rcca.CCA(numCC=n_component, reg = reg, kernelcca=kernel, ktype = ktype, verbose = verbose)
    cca.train([X, Y])
    return cca

def PWCCA(X, Y, reg=0.0, kernel=False, ktype=None, verbose = True):
    c = min(X.shape[1], Y.shape[1])
    cca = do_CCA(X, Y, n_component = c, reg = reg, kernel = kernel, ktype = ktype, verbose = verbose)
    normalized_component, _ = np.linalg.qr(cca.comps[0])
    X = torch.Tensor(X.T).to(device)
    normalized_component = torch.Tensor(normalized_component).to(device)
    #inner = np.matmul(X.T, normalized_component)
    with torch.no_grad():
        inner = torch.matmul(X, normalized_component)
        #print(inner.shape)
        weight = torch.sum(torch.abs(inner), dim = 0).cpu().numpy()
        #print(weight)
    #weight = np.sum(np.abs(inner), axis = 0)
    weighted_cca = np.sum((weight/np.sum(weight))*cca.cancorrs)
    cca_distance = 1 - weighted_cca
    return {'cca':cca, 'cca_dist':cca_distance, 'weighted_cca':weighted_cca}
    
    
def draw_line_chart(list_of_y, list_of_label=None, x = None,
                    xlabel = 'xlabel', ylabel = 'ylabel', xtick = None, ytick = None,
                    xticklabel = None, yticklabel = None,
                    title = None, save_name = None, legend = 'legend',
                    style_list = None, color_num = 1, color_repeat = None, cycle = False, classic = False):
    if classic:
        cmap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
        cmap = cmap[:color_num]
    else:
        cmap = matplotlib.cm.rainbow(np.linspace(0,1,color_num))
    color_count = 0
    
    
    fig = plt.figure(figsize = [7.2,4.8])
    ax = plt.subplot(111)
    if cycle:
        custom_cycler = (cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(style_list)]) +
                         cycler(marker=style_list))
        ax.set_prop_cycle(custom_cycler)
    
    
    if not x:
        x = list(range(len(list_of_y[0])))
    for i, d in enumerate(list_of_y):
        if type(color_repeat)!= type(None):
            if i-color_repeat[0]>=0:
                color_repeat.pop(0)
                color_count += 1
        else:
            color_count = i
            
        if list_of_label:
            label = list_of_label[i]
        else:
            label = ''
            
        if not cycle:
            if type(style_list)!=type(None):
                ax.plot(x, d, style_list[i], label = label, color = cmap[color_count])
            else:
                ax.plot(x, d, 'o-', label = label, color = cmap[color_count])
        else:
            ax.plot(x, d, label = label)

    if type(xtick)!= type(None):
        plt.xticks(xtick)
    if type(ytick)!= type(None):
        plt.yticks(ytick)
    if type(xticklabel)!= type(None):
        ax.set_xticklabels(xticklabel)
    if type(yticklabel)!= type(None):
        ax.set_yticklabels(yticklabel)
    ax.set_xlabel(xlabel, fontsize = 16, horizontalalignment='center', verticalalignment='center')
    ax.set_ylabel(ylabel, fontsize = 16, horizontalalignment='center', verticalalignment='center')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    
    if type(title)!=type(None):
        ax.set_title(title, fontsize = 16)
    if legend == 'legend':
        ax.legend()
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    elif legend == 'color_bar':
        #fig.colorbar(mappable = cmap, ax = ax)#, ticks = np.linspace(0,1,len(list_of_y)))
        print(cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm)
    plt.tight_layout()
    plt.grid()
    #plt.yticks(np.arange(0, 1, step=0.1))
    

    #ax.legend(bbox_to_anchor=(0.75, 0.75))
    

    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels, bbox_to_anchor=(0.4, 0.4))
    if save_name:
        fig.savefig(save_name, bbox_inches = 'tight', pad_inches = 0)
        
        
        
def run_decode_exp(dataloader, model, word = True, layer_err = True, next_cls_err = False, output_layer = False,
            output_inter = False, output_word_embed = False, output_pos_embed = False, skip_connect = None, 
            only_skip = None):
    tqdm_bar = tqdm(dataloader)
    if word:
        output_word_list = {}
    if layer_err:
        layerwirse_err = {}
    if next_cls_err:
        cls_err = {}
    if output_layer or output_inter:
        medium_output_dict = {} 
    #skip = False
    #hidden_norm = {}
    for i in range(model.config.num_hidden_layers+1):
        if word:
            output_word_list['layer_{}'.format(i)] = []
        if layer_err:
            layerwirse_err['layer_{}'.format(i)] = []
        if output_layer:   
            if output_inter:
                medium_output_dict['layer_{}'.format(i)] = {'layer_output':[],
                                                            'ff_preadd':[],
                                                            'attn_output':[],
                                                            'attn_preadd':[]}
            else: medium_output_dict['layer_{}'.format(i)] = {'layer_output':[]}
                
        if next_cls_err:
            cls_err['layer_{}'.format(i)] = 0
        #hidden_norm['layer_{}'.format(i)] = []
    if output_word_embed:
        medium_output_dict['word_embed'] = []
    if output_pos_embed:
        medium_output_dict['pos_embed'] = []
    #oracle = []

    #medium_output_dict = {}  
    count = 0
    for data, seg, att, length, label in tqdm_bar:
        data = data.cuda()
        seg = seg.cuda()
        att = att.cuda()
        bsz = data.shape[0]
        with torch.no_grad():
            if output_inter:
                _, hidden, ff_preAdd, attn_out, attn_preAdd, word_embed, pos_embed  = \
                model(input_ids = data, token_type_ids = seg, attention_mask = att,
                      skip_connect=skip_connect, only_skip = only_skip)
                #print(word_embed.shape)
            elif type(skip_connect)!=type(None) and type(only_skip)!=type(None):
                _, hidden = model(input_ids = data, token_type_ids = seg, attention_mask = att,
                                      skip_connect=skip_connect, only_skip = only_skip)
            else:
                _, hidden = model(input_ids = data, token_type_ids = seg, attention_mask = att)
                
        if output_layer:
            for b in range(bsz):
                medium_output_dict['layer_0']['layer_output'].append(hidden[0][b, 1:length[b]-1].cpu().numpy())
                if output_inter and output_word_embed:
                    medium_output_dict['word_embed'].append(word_embed[b, 1:length[b]-1].cpu().numpy())
                if output_inter and output_pos_embed:
                    medium_output_dict['pos_embed'].append(pos_embed[b, 1:length[b]-1].cpu().numpy())
        for i in range(model.config.num_hidden_layers+1):
            if type(model.cls) == torch.nn.modules.container.ModuleList:
                result = hidden[i].detach()
                for mod in range(len(model.cls)):
                    result = model.cls[mod](result)
                
            else:
                result = model.cls(hidden[i].detach())
                if type(result) == tuple:
                    #warnings.simplefilter('once')
                    warnings.warn("Result type tuple, use index 0")
                    result = result[0]
            #output_tensor = medium_output_dict['layer_{}'.format(i)]
            output_tensor = torch.argmax(result, dim = 2)
            #result = torch.topk
            for b in range(bsz):
                #oracle.append(train_dataset.tokenizer.decode(data[b][1:length[b]-1].tolist()))
                if next_cls_err:
                    cls_err['layer_{}'.format(i)] += (output_tensor[b] == data[b])[1].item()
                    
                if layer_err:
                    word_err = torch.sum( (output_tensor[b] == data[b])[1:length[b]-1] ).item()/(length[b]-2).item()
                    layerwirse_err['layer_{}'.format(i)].append(word_err)
                
                if word:
                    output_list = output_tensor[b].tolist()
                    output_sent = dataloader.dataset.tokenizer.decode(output_list[1:length[b]-1])
                    output_word_list['layer_{}'.format(i)].append(output_sent)
                if output_layer and i!=model.config.num_hidden_layers:
                    medium_output_dict['layer_{}'.format(i+1)]['layer_output'].append(hidden[i][b, 1:length[b]-1].cpu().numpy())
                    if output_inter:
                        medium_output_dict['layer_{}'.format(i+1)]['attn_output'].append(attn_out[i][b, 1:length[b]-1].cpu().numpy())

                        medium_output_dict['layer_{}'.format(i+1)]['ff_preadd'].append(ff_preAdd[i][b, 1:length[b]-1].cpu().numpy())
                        medium_output_dict['layer_{}'.format(i+1)]['attn_preadd'].append(attn_preAdd[i][b, 1:length[b]-1].cpu().numpy())
                    
    return_output = ()
    if word:
        return_output += (output_word_list,)
    if layer_err:
        return_output += (layerwirse_err,)
    if next_cls_err:
        return_output += (cls_err,)
    if output_layer:
        return_output += (medium_output_dict,)
    return return_output


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record) 