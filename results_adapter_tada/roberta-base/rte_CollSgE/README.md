---
language:
- en
license: mit
tags:
- generated_from_trainer
datasets:
- glue
model-index:
- name: rte_CollSgE
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# rte_CollSgE

This model is a fine-tuned version of [roberta-base](https://huggingface.co/roberta-base) on the GLUE RTE dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.6092
- eval_accuracy: 0.6895
- eval_runtime: 9.1404
- eval_samples_per_second: 30.305
- eval_steps_per_second: 3.829
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
