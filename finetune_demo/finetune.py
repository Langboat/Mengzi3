#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The Langboat Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import datasets
import json
import logging
import math
import mmap
import numpy as np
import os
import random
import signal
import sys
import struct
import time
import torch
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datetime import timedelta
from transformers import (SchedulerType, MODEL_MAPPING, default_data_collator, get_scheduler, AutoTokenizer,
                          AutoModelForCausalLM)
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from functools import lru_cache
from plugins.indexed_dataset import MMapIndexedDataset, MMapIndexedDatasetBuilder, make_builder

# use tf32 insteal of fp32
torch.backends.cuda.matmul.allow_tf32 = True

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
_IGNORE_ID = -100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).", )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.")
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.")
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--finetune_embedding_layer",
        action="store_true",
        help="If passed, will only finetune embeddign weights and freezing the rest layers"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False, )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name", )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗  Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.", )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.", )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of gpus for running.", )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ], )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES, )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."), )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.", )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=float,
        default=0.05,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Ratio of warmup steps.", )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.", )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.", )
    parser.add_argument(
        "--enable_recompute",
        action="store_true",
        help="Whether to enable recompute, which can save the needed cuda-memory to speedup training.",
    )
    parser.add_argument(
        "--enable_flash_attn",
        action="store_true",
        help="Whether to enable flash_attn, which can save the needed cuda-memory to speedup training.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."), )
    parser.add_argument(
        '--cache_dir',
        type=str, )
    parser.add_argument(
        '--report_to_dir',
        type=str, )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        # raise ValueError("Need either a dataset name or a training/validation file.")
        pass
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv", "json", "txt", 'jsonl'
            ], "`train_file` should be a csv, json, jsonl or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv", "json", "txt", 'jsonl'
            ], "`validation_file` should be a csv, json, jsonl or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


class SFTDataset(Dataset):
    def __init__(self, indexed_dataset):
        self.indexed_dataset = indexed_dataset

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, idx):
        record = self.indexed_dataset[idx].tolist()
        record_len = len(record)

        input_ids = record[:int(record_len / 2)]
        labels = record[int(record_len / 2):]
        return {'input_ids': input_ids, 'labels': labels}


class Encoder(object):
    def __init__(self, path, max_length=4096):
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, add_bos_token=False, add_eos_token=False)
        self.max_length = max_length
        self.drop_count = 0

    def concat_encode(self, fin, file_path, stop_token_ids):
        input_ids, labels = [], []
        input_ids_temp, labels_temp = [], []
        for line in tqdm(fin, total=get_num_lines(file_path)):
            record = json.loads(line)
            turns = record['conversation']
            source = record['source'] if 'source' in record else ''
            for i in range(len(turns)):
                text = turns[i]['text'].strip()
                if turns[i]['role'] == 'human':
                    text = f"### 指令：\n{text}"
                    text_ids = self.tokenizer(text)['input_ids']
                    if text_ids[0] == 29871:
                        text_ids = text_ids[1:]
                    text_ids = [self.tokenizer.bos_token_id] + text_ids + [self.tokenizer.eos_token_id]
                    input_ids_temp.extend(text_ids)
                    text_labels = [_IGNORE_ID] * len(text_ids)
                    labels_temp.extend(text_labels)
                elif turns[i]['role'] == 'assistant':
                    text = f"### 回复：\n{text}"
                    text_ids = self.tokenizer(text)['input_ids']
                    if text_ids[0] == 29871:
                        text_ids = text_ids[1:]
                    text_ids = [self.tokenizer.bos_token_id] + text_ids + [self.tokenizer.eos_token_id]
                    input_ids_temp.extend(text_ids)
                    text_labels = text_ids.copy()
                    labels_temp.extend(
                        [text_label if text_label not in stop_token_ids else _IGNORE_ID for text_label in text_labels])

            if len(input_ids_temp) > self.max_length:  # 整条对话超过最大长度的情况
                print(text)
                print('整条对话超出最大长度')
                print(len(text), len(input_ids_temp), self.max_length)
                # 丢弃本条
                self.drop_count += 1
                input_ids_temp, labels_temp = [], []
                continue

            if len(input_ids) + len(input_ids_temp) > self.max_length:
                input_ids = [self.tokenizer.bos_token_id] * (self.max_length - len(input_ids)) + input_ids
                labels = [_IGNORE_ID] * (self.max_length - len(labels)) + labels
                yield input_ids, labels, False
                input_ids, labels = [], []

            if not input_ids:
                input_ids, labels = [], []

            input_ids.extend(input_ids_temp)
            labels.extend(labels_temp)
            if len(labels) > self.max_length:
                print('超出最大长度')
                self.drop_count += 1
                input_ids, labels = [], []
            input_ids_temp, labels_temp = [], []

        print('超出最大长度条数：', self.drop_count)
        input_ids = [self.tokenizer.bos_token_id] * (self.max_length - len(input_ids)) + input_ids
        labels = [_IGNORE_ID] * (self.max_length - len(labels)) + labels
        yield input_ids, labels, False


def preprocess_data(args, key, level, dataset_impl='mmap'):
    dataset_path = args.train_file
    tokenizer_path = args.tokenizer_name
    chunk_size = args.block_size
    startup_start = time.time()
    encoder = Encoder(tokenizer_path, chunk_size)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    output_bin_files[key] = args.dataset_name + '.bin'
    output_idx_files[key] = args.dataset_name + '.idx'
    builders[key] = make_builder(
        output_bin_files[key],
        impl=dataset_impl,
        vocab_size=len(encoder.tokenizer) + 1)

    startup_end = time.time()
    print("Time to startup:", startup_end - startup_start)

    stop_token_ids = []
    with open(dataset_path, "r", encoding='utf-8') as fin:
        for input_ids, labels, is_truncated in encoder.concat_encode(fin, dataset_path, stop_token_ids):
            assert len(input_ids) == len(
                labels), "input_ids len(%d), labels len(%d)" % (len(input_ids), len(labels))
            assert len(input_ids) == chunk_size

            if not is_truncated:
                insert_ids = input_ids.copy()
                insert_ids.extend(labels)
                builders[key].add_item(torch.IntTensor(insert_ids))
                builders[key].end_document()

    print("Done! Now finalizing.")
    builders[key].finalize(output_idx_files[key])


def main():
    args = parse_args()
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=180000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[kwargs],
        log_with=args.report_to,
        mixed_precision='bf16',
        project_dir=args.report_to_dir)

    # preprocess data
    level = "document"
    key = "text"
    output_prefix = 'example'
    args.dataset_name = "{}_{}_{}".format(output_prefix, key, level)
    if accelerator.is_main_process:
        preprocess_data(args, key, level)
    accelerator.wait_for_everyone()

    if args.report_to != "" and args.report_to is not None:
        _train_config = {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_warmup_steps": args.num_warmup_steps
        }
        accelerator.init_trackers(
            config=_train_config, project_name="mengzi2_13b_finetune")

    accelerator.state.deepspeed_plugin.deepspeed_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    accelerator.state.deepspeed_plugin.deepspeed_config[
        'train_batch_size'] = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, )
    logger.info(accelerator.state, main_process_only=True)
    logger.info(f'Args: {args}')

    # only local main process can print warnings
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    idxed_train_dataset = MMapIndexedDataset(args.dataset_name)
    train_dataset = SFTDataset(idxed_train_dataset)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=args.use_fast_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=args.use_fast_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        raise ValueError("args.model_name_or_path can not be None")

    logger.info(model)
    logger.info(model.named_parameters())
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size)

    ## add recompute
    if args.enable_recompute:
        model.gradient_checkpointing_enable()

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        }
    ]
    logger.info('optimizer:')
    logger.info(optimizer_grouped_parameters)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps *
        args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps *
        args.gradient_accumulation_steps
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is None:
        checkpointing_steps = 0.05

    checkpointing_steps = int(len(train_dataloader) * checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Num steps per epoch = {len(train_dataloader)}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.print(
            f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)

        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            print(f'training_difference -> {training_difference}', flush=True)
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace(
                "step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch
    total_loss = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                # print(f'batch -> {batch}')
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.report_to != "" and args.report_to is not None:
                    if accelerator.is_main_process:
                        total_loss = loss.detach().float().item()
                        accelerator.log({
                            "tran_loss": total_loss,
                            "epoch": epoch,
                            "step": step,
                            "learning_rate": lr_scheduler.get_last_lr()[0]
                        }, step=completed_steps)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

    if args.report_to != "" and args.report_to is not None:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
