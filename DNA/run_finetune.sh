export USE_TORCH=1
task=$1
epoch="1"
pretrain=$2
model=bert-base-uncased
seed=$3
#mkdir ./save_model/$task
#mkdir ./save_model/$task/$seed
python finetune.py --task $task \
	--type $pretrain \
	--seed $seed \
	--model $model \
	--logdir ./log/$task/$model/$seed/$pretrain \
    --datadir ./data \
	-b 8 \
	-e $epoch \
    --save_step 1000 \
    --savedir ./save_model/$task/$model/$seed \
    --shift_table ../assign_token/$model/table_seed${seed}.pkl \
	--n_gpu 1

