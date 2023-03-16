export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

#mnli qnli rte sst2 stsb qqp
TASKS="cola rte"
for MODEL_NAME in roberta-base #bert-base-uncased
do
    for TASK_NAME in $TASKS
    do
	ADAPTER_ADDRESS=WillHeld/pfadapter-${MODEL_NAME}-${TASK_NAME}
	TADA_ADDRESS=WillHeld/pfadapter-${MODEL_NAME}-tada-adv
	echo $TASK_NAME
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_aave \
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --dialect="aave" \
	       --load_adapter $ADAPTER_ADDRESS \
	       --load_lang_adapter $TADA_ADDRESS-aave \
	       --do_eval

	# python run_glue_adapterhub.py \
	#        --model_name_or_path $MODEL_NAME \
	#        --task_name $TASK_NAME \
	#        --output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_CollSgE \
	#        --overwrite_output_dir \
	#        --adapter_config pfeiffer \
	#        --train_adapter \
	#        --dialect="CollSgE" \
	#        --load_adapter $ADAPTER_ADDRESS \
	#        --load_lang_adapter $TADA_ADDRESS-CollSgE \
	#        --do_eval

	# python run_glue_adapterhub.py \
	#        --model_name_or_path $MODEL_NAME \
	#        --task_name $TASK_NAME \
	#        --output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_IndE \
	#        --overwrite_output_dir \
	#        --adapter_config pfeiffer \
	#        --train_adapter \
	#        --dialect="IndE" \
	#        --load_adapter $ADAPTER_ADDRESS \
	#        --load_lang_adapter $TADA_ADDRESS-IndianEnglish \
	#        --do_eval

	# python run_glue_adapterhub.py \
	#        --model_name_or_path $MODEL_NAME \
	#        --task_name $TASK_NAME \
	#        --output_dir ./results_adapter_tada/$MODEL_NAME/${TASK_NAME}_NgE \
	#        --overwrite_output_dir \
	#        --adapter_config pfeiffer \
	#        --train_adapter \
	#        --dialect="NgE" \
	#        --load_adapter $ADAPTER_ADDRESS \
	#        --load_lang_adapter $TADA_ADDRESS-NigerianEnglish \
	#        --do_eval
    done
done
