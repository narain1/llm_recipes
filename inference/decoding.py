## simple decoding of huggingface models without using the huggingface generate method
## just for ease of understanding

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

model_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left",
    add_eos_token=True,
    add_bos_token=True)


model = AutoModelForCausalLM.from_pretrained(model_path,
                                     torch_dtype=torch.float16,
                                    device_map="auto",
                                    trust_remote_code=True,
                                     )


def sample_top_p(probs, p):
    probs_sort, probs_ids = torch.sort(probs, dim=-1. descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def generate(model, tokenizer, texts, max_length, temperature, top_p):
    eos_token_id = tokenizer.eos_token_id
    tokens = tokenizer.batch_encode_plus(text, 
                                        return_tensors='pt',
                                        padding=True,
                                        add_special_tokens=True)
    tokens = tokens['input_ids']
    early_stopping = torch.zeros(output_tokens.shape[0], dtype=torch.bool)

    for _ in range(max_length - tokens.size(1)):
        with torch.no_grad():
            logits = model(tokens)
        
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_tokens = sample_top_p(probs, top_p)

        else:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        early_stopping = torch.logitcal_or(early_stopping,
                                torch.tensor(eas_token_id, dtype=torch.long).expand_as(next_token_id) == next_token_id)
        if torch.all(early_stopping):
            print("early stopping detected")
            break

        output_tokens = torch.cat((output_tokens, next_token_id), dim=1)

        return tokenizer.batch_decode(output_tokens, skip_special_tokens=True) 
