export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

HF_ORG="WillHeld"

TASKS="qqp" #"cola mnli qnli rte sst2 stsb" #qqp
for MODEL_NAME in roberta-base
do
    for TASK_NAME in $TASKS
    do
	ADAPTER_ADDRESS=SALT-NLP/pfadapter-${MODEL_NAME}-${TASK_NAME}-combined-value
	echo $ADAPTER_ADDRESS
	python run_glue_adapterhub.py \
	           --model_name_or_path $MODEL_NAME \
	           --task_name $TASK_NAME \
	           --output_dir ./results_adapter_combo/$MODEL_NAME/$TASK_NAME \
	           --overwrite_output_dir \
	           --adapter_config pfeiffer \
	           --train_adapter \
	           --load_adapter ./results_combo_adapt/roberta-base/qqp/checkpoint-326500//qqp \
	           --do_eval

	python run_glue_adapterhub.py \
	           --model_name_or_path $MODEL_NAME \
	           --task_name $TASK_NAME \
	           --output_dir ./results_adapter_combo/$MODEL_NAME/${TASK_NAME}_aave \
	           --overwrite_output_dir \
	           --adapter_config pfeiffer \
	           --train_adapter \
	           --dialect="aave" \
	           --load_adapter ./results_combo_adapt/roberta-base/qqp/checkpoint-326500//qqp \
	           --do_eval
    done
done
