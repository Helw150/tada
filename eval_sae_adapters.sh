export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')


TASKS="cola mnli qnli rte sst2 stsb qqp"
for MODEL_NAME in roberta-base bert-base-uncased
do
    for TASK_NAME in $TASKS
    do
	ADAPTER_ADDRESS=WillHeld/pfadapter-${MODEL_NAME}-${TASK_NAME}
	echo $TASK_NAME
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results_adapter_sae/$MODEL_NAME/$TASK_NAME \
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --load_adapter $ADAPTER_ADDRESS \
	       --do_predict

	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results_adapter_sae/$MODEL_NAME/${TASK_NAME}_aave \
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --dialect="aave" \
	       --load_adapter $ADAPTER_ADDRESS \
	       --do_predict
	
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results_adapter_sae/$MODEL_NAME/${TASK_NAME}_IndE\
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --dialect="IndE" \
	       --load_adapter $ADAPTER_ADDRESS \
	       --do_predict

	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results_adapter_sae/$MODEL_NAME/${TASK_NAME}_CollSgE\
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --dialect="CollSgE" \
	       --load_adapter $ADAPTER_ADDRESS \
	       --do_predict

	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results_adapter_sae/$MODEL_NAME/${TASK_NAME}_NgE\
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --dialect="NgE" \
	       --load_adapter $ADAPTER_ADDRESS \
	       --do_predict
	
    done
done
