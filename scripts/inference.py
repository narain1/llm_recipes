from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import gc
import itertools

def longest_common_prefix(strs):
    if not strs:
        return ""

    shortest = min(strs, key=len)

    for i, char in enumberate(shortes):
        for other in strs:
            if other[i] != char:
                return shortes[:i]

    return shortest


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--device", type=str, required=True)
    arg.add_argument("--model_name", type=str, required=True)
    arg.add_argument("--quantization", type=int, required=True)
    arg.add_argument("--model_type", type=int, required=True)
    arg.add_argument("--test_file", type=str, required=True)
    args = arg.parse_args()


    if args.device == "auto":
        DEV_MAP = "auto"
        DEV = "cuda:0"
    else:
        DEV_MAP = {"": args.device}
        DEV = args.device

    llm_backbone = args.model_name

    test = pd.read_parquet(args.test_file).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(
        llm_backbone,
        use_fast=True,
        trust_remote_code=True,
        padding_side="right",
        truncation_side="left",
    )

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token

    if args.quantization == 0:
        quantization_config = None
    elif args.quantization == 1:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=0.0
        )

