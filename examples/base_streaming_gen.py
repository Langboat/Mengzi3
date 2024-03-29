#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The Langboat Inc. team. All rights reserved.
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


import argparse
import numpy as np
import random
import torch
from rich import print
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers_stream_generator import init_stream_support

init_stream_support()


def parse_args():
    parser = argparse.ArgumentParser(description="test base model")
    parser.add_argument(
        "--tokenizer", type=str, default='Langboat/Mengzi3-13B-Base')
    parser.add_argument(
        "--model", type=str, default='Langboat/Mengzi3-13B-Base')
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)

    return parser.parse_args()


# set all seeds to make results reproducible
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate(inputs):
    generator = model.generate(
        **inputs,
        max_new_tokens=512,
        min_new_tokens=1,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        top_k=1,
        num_return_sequences=1,
        repetition_penalty=1.1,
        do_stream=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    for token in generator:
        word = tokenizer.decode(token)
        print(word, end="")
    print("\n")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.random_seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model).to("cuda:%d" % args.device_id)
    model.half()
    model.eval()
    model = torch.compile(model)

    while True:
        user_input = input("输入你的prompt: ")
        inputs_dict = tokenizer(user_input, return_tensors="pt").to("cuda:%d" % args.device_id)
        generate(inputs_dict)
