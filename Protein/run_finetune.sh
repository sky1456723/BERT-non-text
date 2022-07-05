seed=$1
task=$2
model=$3
if [ "$task" = "stability" ]
then
	epoch=1
else
	epoch=1
fi
#epoch=$4
#mkdir ./save_model/$task/$seed
python finetune.py --task $task \
    --type pretrain \
    --seed $seed \
    --model $model \
    --logdir ./log/$task/$model/$seed \
    -b 16 \
    -e $epoch \
    --n_gpu 1 \
    --save_step 3000 \
    --savedir ./save_model/$task/$model/$seed \
    --shift_table ../assign_token/$model/table_seed${seed}.pkl
