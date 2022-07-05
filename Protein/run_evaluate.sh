export USE_TORCH=1
task=$1
pretrain=$2
split=$3
step=0
seed=$4
python evaluate.py --task $task \
    --split $split \
    --step $step \
    --model bert-base-uncased \
    -b 64 \
    --type scratch \
    --shift_table ../assign_token/bert-base-uncased/table_seed${seed}.pkl \
    --state_dict ./save_model/${task}/bert-base-uncased/$seed/${task}_bert-base-uncased_${pretrain}_seed100_table__670.pkl \
    --logdir ./log/$task/bert-base-uncased/$seed

#${pretrain}_dev_best.pkl \
