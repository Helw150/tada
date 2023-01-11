export CUDA_VISIBLE_DEVICES=1

HF_ORG="WillHeld"

for MODEL_NAME in roberta-base #bert-base-uncased #roberta-base
do
    MODEL=pfadapter-${MODEL_NAME}-tada-values
    echo $MODEL
    py contrastive_adapter.py \
       --do_eval=False \
       --model_name_or_path $MODEL_NAME \
       --dataset_name "super_glue" \
       --dataset_config_name "wic" \
       --text_column "sentence1" \
       --dialect "aave" \
       --max_train_samples 1000 \
       --max_eval_samples 100 \
       --output_dir ./tada_train_layer0/$MODEL_NAME \
       --max_seq_length 128 \
       --per_device_train_batch_size 16 \
       --learning_rate 5e-4 \
       --num_train_epochs 30 \
       --push_to_hub False \
       --push_adapter_to_hub \
       --adapter_org_id $HF_ORG \
       --adapter_repo_id $MODEL \
       --adapter_config pfeiffer+inv \
       --evaluation_strategy "steps" \
       --logging_steps 50 \
       --eval_steps 125 \
       --save_steps 125 \
       --save_total_limit 1 \
       --overwrite_output_dir \
       --logging_steps 50 \
       --use_auth_token \
       --do_train
done
