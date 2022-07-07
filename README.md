This is the code for the EMNLP 2021 findings paper [Is BERT a Cross-Disciplinary Knowledge Learner? A Surprising Finding of Pre-trained Models' Transferability](https://arxiv.org/abs/2103.07162).

# Envrionment
All the package version are listed in the `requirements.txt` files. The basic packages used are the following:
```
transformers=3.1.0
torch=1.6.0
numpy=1.19.1
scipy=1.5.2
sklearn
pandas
matplotlib
cycler
h5py
protobuf=3.20.0
tensorboardX=2.1
tqdm=4.48.2
pretty_midi=0.2.9
```

Recommandation: You can pull the docker image `sky1456723/handover:latest` (or pull `sky1456723/handover:dna` and `pip install -r requirements.txt` in the docker container) to prepare the environment.

# Data preparation
## GLUE
To use the GLUE dataset, we use the script of the example script of the **transformers** library: 

https://github.com/huggingface/transformers 

Please use the `download_glue_data.py` to download the glue dataset. The file is copied and modified from GLUE dataset's repo: https://github.com/nyu-mll/GLUE-baselines (I add one entry in the python dictionary in line 33.).
```
python download_glue_data.py --data_dir /path/to/glue --tasks all
```
and 
```
export GLUE_DIR=/path/to/glue 
```
before run our script.

## Protein classification
Please do the following under the `Protein` directory:
1. Please download the data from the site: http://ailab.snu.ac.kr/PLUS/ (Because the website is not stable, I also make a backup on battleship. The datasets are put in the folder `/livingrooms/jimmykao1453/backup_protein_data`.)
2. Unzip the data (.fa files) to the `data` directory
3. Run `save_feature.py` to preprocess the input feature and save it to `preprocess_input/TASK_NAME/` (please `mkdir` these folders by yourself.), TASK_NAME is the corresponding task name. The input arguments of `save_feature.py` are:
```
--task The downstream task name
--model The pre-trained model name, default is bert-base-uncased
--savedir The directory to save
--split The train/dev/test split to preprocess
--ratio To control the training dataset size (for the generalization ability experiments).
```
The `--task` could be `localization`, `fluorescence`, or `stability`.

## DNA classification
Please do the following under the `DNA` directory:
1. Please clone the git repository:https://github.com/Doulrs/Hilbert-CNN
2. Run `preprocess.py` and `preprocess_splice.py` to preprocessing the input data. The input arguments are similar to `save_feature.py` in the `Protein` directory.

The `--task` of `preprocess.py` could be `H3`, `H3K9ac`, or `H4`.

## Music 
Please do the following under the `Music` directory:
1. Download the MAESTRO-v1 dataset from:https://magenta.tensorflow.org/datasets/maestro#download (downloading the midi data and the metadata is enough.)
2. Unzip the data under the `Music/raw_data` directory.
3. Run `preprocess.py` to preprocess the input data and save it to the `Music/data/maestro-v1` directory.

# Fine-tune and evaluate

Note: I only prepare the token mapping table files for random seed 100-109 under `assign_token` folder. If you would like to use other seeds, please run `gen_table.py` first to generate the mapping table. The input arguments of `gen_table.py` are
```
--model The model to use (for tokenizer).
--seed The random seed to use.
--save_dir The directory to save the generated file.
```

## GLUE
We modified the **transformers** library v.3.1.0. The modified source code are in the `transformers` directory under the `GLUE` dataset. Please use the modified library but not the original one. 

We add several new input arguments to `run_glue.py`:
1. pretrain_ckpt: used for the experiment of different checkpoints.
2. scratch: used for the model trained from scratch.
3. rand_embed: used for the ablation study of randomly initialized embedding.
4. shift: the constant `c` for the "shift c" setting.
5. random_shift: use for the `random_shift` setting.

The other input arguments are the same as the huggingface example. Please use the `run_glue.py` script to fine-tune and evaluate the models. We offer an example shell script `run_hf_glue.sh` to run the script. Please use `source` but not `sh` to execute this shell script. (The script will do fine-tuning and then do evaluation.)

## Protein Classification
The code about protein classification is modified from the code provided at https://github.com/mswzeus/PLUS/
To run the experiment, please run `finetune.py` and `evaluate.py`, the explaination of each input argument is written in the help message of `finetune.py`. We offer an example to run the script in `run_finetune.sh` and `run_evaluate.sh`.

\*Remark: Please using the "--shift" argument or the "--shift_table" and the shift_table files under the `assign_token` directory to avoid using the unused tokens. This is necessary to reproduce the performance.

\*Remark: For the dataloader, we set `drop_last=True` to prevent some error when the last batch contains only 1 data. It can be set to `False`if there is no such problem.

## DNA classification
To run the experiment, please run `finetune.py` and `evaluate.py`, the input arguments are similar to the `finetune.py` in the `Protein` directory. We offer an example to run the script in `run_finetune.sh` and `run_evaluate.sh` under the `DNA` directory.

## Music
To run the experiment, please run `finetune.py` and `evaluate.py`, the input arguments are similar to the `finetune.py` in the `Protein` directory. We offer an example to run the script in `run_finetune.sh` and `run_evaluate.sh` under the `Music` directory, too.

\* Remark: Please using the "--shift" argument or the shift_table file under the `data` directory to avoid using the unused tokens.

# Logging
All the results are loggeg by tensorboardX, including training loss, dev loss, test loss, and the corresponding evaluation metric. The evaluation results are logged at the time step corresponding to the fine-tuning steps.

