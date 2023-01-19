export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

HF_ORG="WillHeld"

TASKS="cola mnli qnli rte sst2 stsb qqp"
for MODEL_NAME in roberta-base
do
    for TASK_NAME in $TASKS
    do
	MODEL=$HF_ORG/${MODEL_NAME}-${TASK_NAME}
	echo $MODEL
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results_train_combined/$MODEL_NAME/$TASK_NAME \
	       --max_seq_length 128 \
	       --per_device_train_batch_size 16 \
	       --learning_rate 2e-5 \
	       --weight_decay 0.1 \
	       --warmup_ratio 0.06 \
	       --num_train_epochs 10 \
	       --overwrite_output_dir \
	       --do_train \
	       --save_total_limit 1 \
	       --push_to_hub True \
	       --hub_model_id $MODEL \
	       --hub_private_repo \
	       --use_auth_token
    done
done
