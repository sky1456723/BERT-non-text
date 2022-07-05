export USE_TORCH=1
task="maestro-v1"
epoch=1
seed=$1
model=bert-base-uncased #$2
#mkdir ./save_model/$task
#mkdir ./save_model/$task/$seed
python finetune.py --task $task \
	--type pretrain \
    --seed $seed \
    --model $model \
    --logdir ./log/$task/$model/$seed \
    --datadir ./data \
    -b 32 \
    -a 1 \
    -e $epoch \
    --save_step 1000 \
    --n_gpu 1 \
    --savedir ./save_model/$task/$model/$seed \
    --shift_table ../assign_token/$model/table_seed${seed}.pkl
