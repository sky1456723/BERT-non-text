export GLUE_DIR=./glue_data
export CUDA_VISIBLE_DEVICES=0
#export TASK_NAME=$1
#dir=$1
#run_name="${TAKS_NAME}_${dir}"
seeds=(100) #(100 101 102)
shift=$1
#tasks="MNLI QNLI QQP SST-2 STS-B MRPC RTE CoLA"
tasks="STS-B"
for task in $tasks
do
	for seed in ${seeds[@]}
	do 
		python run_glue.py \
          --config_name bert-base-uncased \
		  --model_name_or_path bert-base-uncased \
		  --tokenizer_name bert-base-uncased \
		  --task_name ${task} \
		  --do_train \
		  --do_eval \
		  --data_dir $GLUE_DIR/${task} \
		  --max_seq_length 128 \
		  --per_device_train_batch_size 32 \
		  --per_device_eval_batch_size 32 \
		  --learning_rate 1e-5 \
		  --max_grad_norm 1e6 \
		  --num_train_epochs 3.0 \
		  --logging_steps 100 \
          --save_steps 20000 \
		  --output_dir run_result/${task}/${shift}/rand_shift2/scratch/rand${seed} \
		  --logging_dir runs/${task}/${shift}/rand_shift2/scratch/rand${seed} \
		  --seed ${seed} \
		  --shift ${shift} \
          --rand_embed \
          --random_shift \
          --scratch

        #python run_glue_310.py \
		#  --model_name_or_path bert-base-uncased \
		#  --tokenizer_name bert-base-uncased \
		#  --task_name ${task} \
		#  --do_train \
		#  --do_eval \
		#  --data_dir $GLUE_DIR/${task} \
		#  --max_seq_length 128 \
		#  --per_device_train_batch_size 16 \
		#  --per_device_eval_batch_size 16 \
		#  --learning_rate 1e-5 \
		#  --max_grad_norm 1e6 \
		#  --num_train_epochs 3.0 \
		#  --logging_steps 100 \
        #  --save_steps 20000 \
		#  --output_dir run_result/${task}/${shift}/test/rand${seed} \
		#  --logging_dir runs/${task}/${shift}/test/rand${seed} \
		#  --seed ${seed} \
		#  --shift ${shift} \
        #  --scratch

    done
done 
