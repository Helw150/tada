export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

HF_ORG="WillHeld"

TASKS="cola mnli qnli rte sst2 stsb qqp"
for MODEL_NAME in roberta-base
do
    for TASK_NAME in $TASKS
    do
	ADAPTER_ADDRESS=$HF_ORG/pfadapter-${MODEL_NAME}-${TASK_NAME}
	echo $ADAPTER_ADDRESS
	py run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results_combo_adapt/$MODEL_NAME/$TASK_NAME \
	       --max_seq_length 128 \
	       --per_device_train_batch_size 16 \
	       --learning_rate 1e-4 \
	       --weight_decay 0.1 \
	       --warmup_ratio 0.06 \
	       --num_train_epochs 15 \
	       --overwrite_output_dir \
	       --do_train \
	       --push_adapter_to_hub True \
	       --adapter_org_id $HF_ORG \
	       --adapter_repo_id $ADAPTER_ADDRESS \
	       --save_total_limit 1 \
	       --hub_private_repo \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --use_auth_token
    done
done
