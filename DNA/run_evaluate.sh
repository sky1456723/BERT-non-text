export USE_TORCH=1
task=$1
step=842
seed=100
model=$2
split=$3
python evaluate.py --task $task \
    --split $split \
    --step $step \
    --model $model \
    -b 64 \
    --type pretrain \
    --state_dict ./save_model/$task/bert-base-uncased/$seed/${task}_${model}_pretrain_seed${seed}_table__${step}.pkl \
    --logdir ./log/$task/model/$seed \
    --datadir ./data \
    --shift_table ../assign_token/bert-base-uncased/table_seed${seed}.pkl
