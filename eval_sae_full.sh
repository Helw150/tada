export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

HF_ORG="WillHeld"

TASKS="cola" #mnli qnli rte qqp sst2 stsb"
for MODEL_NAME in roberta-base
do
    for TASK_NAME in $TASKS
    do
	MODEL=$HF_ORG/${MODEL_NAME}-${TASK_NAME}
	echo $MODEL	
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL \
	       --task_name $TASK_NAME \
	       --output_dir ./results_full_sae/$MODEL/$TASK_NAME \
	       --overwrite_output_dir \
	       --do_eval \
	       --hub_model_id=$MODEL \
	       --hub_private_repo \
	       --use_auth_token

	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL \
	       --task_name $TASK_NAME \
	       --output_dir ./results_full_sae/$MODEL/${TASK_NAME}_aave \
	       --overwrite_output_dir \
	       --dialect="aave" \
	       --do_eval \
	       --hub_model_id=$MODEL \
	       --hub_private_repo \
	       --use_auth_token
    done
done
