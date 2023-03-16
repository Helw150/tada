---
language:
- en
license: mit
tags:
- generated_from_trainer
datasets:
- glue
model-index:
- name: stsb_CollSgE
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stsb_CollSgE

This model is a fine-tuned version of [roberta-base](https://huggingface.co/roberta-base) on the GLUE STSB dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.4305
- eval_pearson: 0.8995
- eval_spearmanr: 0.8972
- eval_combined_score: 0.8984
- eval_runtime: 12.6858
- eval_samples_per_second: 118.242
- eval_steps_per_second: 14.82
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
