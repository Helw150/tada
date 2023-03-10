#!/usr/bin/env python
# coding=utf-8

"""
Aligning Embeddings using contrastive learning on aligned data
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset

import transformers
from transformers.adapters.modeling import Adapter
from models import (
    BartAdapterModelForAlignment,
    RobertaAdapterModelForAlignment,
    BertAdapterModelForAlignment,
)
from transformers import (
    AdapterConfig,
    AdapterTrainer,
    AutoAdapterModel,
    AutoTokenizer,
    HfArgumentParser,
    LoRAConfig,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import pickle as pkl

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from multivalue.src.Dialects import *

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )

    dialect: Optional[str] = field(
        default=None,
        metadata={"help": "the directory where VALUE datasets will be saved"},
    )

    load_dialect_from_hub: bool = field(
        default=False,
        metadata={"help": "Whether to load the dialect dataset from Huggingface Hub"},
    )

    push_adapter_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the Adapter to HuggingFace ModelHub"},
    )

    adapter_org_id: Optional[str] = field(
        default=None,
        metadata={"help": "Organization to contain AdapterHub repo"},
    )

    adapter_repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "the Hub Repo name for model to push"},
    )

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory containing input files."}
    )

    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the datasets to do alignment on"},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "json", "`validation_file` should be a json file."


def collate_fn(examples):
    original_embedding = torch.stack(
        [
            torch.tensor(example["original_embedding"]).reshape(128, -1)
            for example in examples
        ]
    )

    input_ids = torch.tensor(
        [example["input_ids"] for example in examples], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [example["attention_mask"] for example in examples], dtype=torch.long
    )
    return {
        "original_embedding": original_embedding,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            MultiLingAdapterArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
    ) = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventualy continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 5. Load pretrained model, tokenizer, and feature extractor
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if "roberta" in model_args.model_name_or_path:
        model = RobertaAdapterModelForAlignment.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif "bart" in model_args.model_name_or_path:
        model = BartAdapterModelForAlignment.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = BertAdapterModelForAlignment.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    config = model.config

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    text_column = data_args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
        )

    transformed_column = "transformed_" + text_column

    def dialect_transform_factory(dialect_name):
        def dialect_transform(examples):
            dialect = None
            if dialect_name == "aave":
                dialect = AfricanAmericanVernacular(morphosyntax=True)
            elif dialect_name == "IndianEnglish":
                dialect = IndianDialect(morphosyntax=True)
            elif dialect_name == "CollSgE":
                dialect = ColloquialSingaporeDialect(morphosyntax=True)
            elif dialect_name == "NigerianEnglish":
                dialect = NigerianDialect(morphosyntax=True)

            original_text = examples[text_column] + examples["sentence2"]
            transformed_text = [
                dialect.convert_sae_to_dialect(example) for example in original_text
            ]
            del examples
            examples = {}
            examples["original"] = original_text
            examples["transformed"] = transformed_text

            return examples

        return dialect_transform

    if not data_args.load_dialect_from_hub:
        dialect_transform = dialect_transform_factory(data_args.dialect)
        dataset = dataset.map(
            dialect_transform,
            batched=True,
            remove_columns=[
                column_name
                for column_name in column_names
                if column_name not in ["original", "transformed"]
            ],
            load_from_cache_file=not data_args.overwrite_cache,
            num_proc=1,
            desc="Transform Dataset Using Dialect Transformations",
        )

    else:
        dataset = load_dataset(
            "WillHeld/TADA",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_text(examples):
        result = tokenizer(
            examples["transformed"],
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        return result

    # Generate Fixed Original Embeddings
    def embed_text(examples):
        result = tokenizer(
            examples["original"],
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        batch_size = 128
        original_embeddings = []
        for i in range(0, len(result["input_ids"]), batch_size):
            input_ids = torch.tensor(result["input_ids"][i : i + batch_size]).cuda()
            attention_mask = torch.tensor(
                result["attention_mask"][i : i + batch_size]
            ).cuda()
            hidden_mat = model.produce_original_embeddings(input_ids, attention_mask)
            original_embeddings.extend(
                [embedding.flatten() for embedding in torch.split(hidden_mat, 1)]
            )
        examples["original_embedding"] = original_embeddings
        return examples

    model = model.cuda()
    dataset = dataset.map(
        tokenize_text,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    ).map(
        embed_text,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Original Embeddings for Untransformed Input",
    )

    adapter_config = AdapterConfig.load(
        adapter_args.adapter_config,
        non_linearity=adapter_args.adapter_non_linearity,
        reduction_factor=adapter_args.adapter_reduction_factor,
    )

    if adapter_args.load_adapter:
        model.load_adapter(
            adapter_args.load_adapter,
            config=adapter_config,
            is_parallel=True,
            load_as="tada_aave",
        )
    else:
        print(len(list(model.iter_layers())))
        print("test")
        model.add_adapter("tada_aave", config=adapter_config)
        model.train_adapter(["tada_aave"], train_embeddings=True)
    model.set_active_adapters(["tada_aave"])

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        validation_split = (
            "validation_matched"
            if data_args.dataset_config_name == "mnli"
            else "validation"
        )
        if validation_split not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset[validation_split]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = dataset["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(test_dataset), data_args.max_eval_samples)
            test_dataset = test_dataset.select(range(max_eval_samples))

    # 8. Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Write Training Stats and push to hub.
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "embedding-alignment",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    elif data_args.push_adapter_to_hub:
        model.push_adapter_to_hub(
            data_args.adapter_repo_id,
            "tada_aave",
            organization=data_args.adapter_org_id,
            private=training_args.hub_private_repo,
            use_auth_token=model_args.use_auth_token,
            adapter_card_kwargs=kwargs,
            datasets_tag="glue",
        )
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
