---
language:
- en
license: mit
tags:
- generated_from_trainer
datasets:
- glue
model-index:
- name: qqp_NgE
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qqp_NgE

This model is a fine-tuned version of [roberta-base](https://huggingface.co/roberta-base) on the GLUE QQP dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.2784
- eval_accuracy: 0.8817
- eval_f1: 0.8492
- eval_combined_score: 0.8655
- eval_runtime: 131.3021
- eval_samples_per_second: 307.916
- eval_steps_per_second: 38.491
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.24.0
- Pytorch 1.12.1+cu102
- Datasets 2.6.1
- Tokenizers 0.11.6
