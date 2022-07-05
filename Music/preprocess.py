import pretty_midi
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str)
args = parser.parse_args()

model_name = args.model
task = 'maestro-v1'
csv = pd.read_csv(f'./raw_data/{task}.0.0.csv')
                  
composer = csv['canonical_composer']
split = csv['split']
midi_filename = csv['midi_filename']
composer2id = {}
idx = 0
for k in range(len(composer)):
    if composer[k] not in composer2id.keys():
        composer2id[composer[k]] = idx
        idx += 1
        
def get_pitch(midi_data):
    note_list = []
    instrument = midi_data.instruments
    notes = instrument[0].notes
    i = 0
    while (i+1)*128 < len(notes):
        note = sorted(notes[i*128:(i+1)*128], key=lambda x:x.start)
        note = [ n.pitch for n in note ]
        note_list.append(note)
        i += 1
    return note_list

train_data = []
train_label = []
dev_data = []
dev_label = []
test_data = []
test_label = []
for i in tqdm(range(len(composer))):
    midi_data = pretty_midi.PrettyMIDI(f'./raw_data/{task}.0.0/{midi_filename[i]}')
    pitch = get_pitch(midi_data)
    label = composer2id[composer[i]]
    if split[i] == 'train':
        train_data.extend(pitch)
        train_label.extend([label]*len(pitch))
    elif split[i] == 'validation':
        dev_data.extend(pitch)
        dev_label.extend([label]*len(pitch))
    elif split[i] == 'test':
        test_data.extend(pitch)
        test_label.extend([label]*len(pitch))
    else:
        raise NotImplementedError
                  
train_t = (torch.Tensor(train_data)+128).long()
dev_t = (torch.Tensor(dev_data)+128).long()
test_t = (torch.Tensor(test_data)+128).long()
torch.save(train_t, f'./data/{task}/{task}_{model_name}_train_data.pkl')
torch.save(torch.LongTensor(train_label), f'./data/{task}/{task}_{model_name}_train_label.pkl')
torch.save(dev_t, f'./data/{task}/{task}_{model_name}_dev_data.pkl')
torch.save(torch.LongTensor(dev_label), f'./data/{task}/{task}_{model_name}_dev_label.pkl')
torch.save(test_t, f'./data/{task}/{task}_{model_name}_test_data.pkl')
torch.save(torch.LongTensor(test_label), f'./data/{task}/{task}_{model_name}_test_label.pkl')
torch.save(composer2id, f'./data/{task}/composer2id_map.pkl')


dev_data = []
dev_label = []
test_data = []
test_label = []
for i in tqdm(range(len(composer))):
    if split[i] == 'train':
        continue
    midi_data = pretty_midi.PrettyMIDI(f'./raw_data/{task}.0.0/{midi_filename[i]}')
    pitch = get_pitch(midi_data)
    label = composer2id[composer[i]]
    if split[i] == 'validation':
        dev_data.append((torch.Tensor(pitch)+128).long())  
        dev_label.extend([label]*1)
    elif split[i] == 'test':
        test_data.append((torch.Tensor(pitch)+128).long())
        test_label.extend([label]*1)
    else:
        raise NotImplementedError
        
torch.save(dev_data, f'./data/{task}/{task}_{model_name}_dev_data.pkl')
torch.save(torch.LongTensor(dev_label), f'./data/{task}/{task}_{model_name}_dev_label.pkl')
torch.save(test_data, f'./data/{task}/{task}_{model_name}_test_data.pkl')
torch.save(torch.LongTensor(test_label), f'./data/{task}/{task}_{model_name}_test_label.pkl')
