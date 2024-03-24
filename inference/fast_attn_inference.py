## batch decoding
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm.auto import tqdm, trange
import pandas as pd
import numpy as np
import os


os.environ['TRANSFORMERS_CACHE'] = '/scratch/npattab1/hf_cache'
os.environ['HF_HOME'] = '/scratch/npattab1/hf_cache'
access_token = "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"
model_id = "google/gemma-7b-it"

df = pd.read_csv('train.csv')
df = df.fillna('')
bs = 8
df = df.sample(1000).reset_index(drop=True)


prompts = []
for k, row in df.iterrows():
    text = row['instruction'] + (f'\ntext: {row["input"]}' if row['input'] != '' else '')
    prompts.append(text)

print(prompts[:5])

tokenizer = AutoTokenizer.from_pretrained(model_id,  cache_dir='/scratch/npattab1/llms/', padding_side='left')
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir='/scratch/npattab1/llms/',
    attn_implementation="flash_attention_2",
)

generated_text = []
with torch.inference_mode():
    for k, idx in enumerate(trange(0, len(prompts), bs)):
        tokens = tokenizer(prompts[idx: idx+bs],  return_tensors="pt", padding='longest').to('cuda')
        output = model.generate(**tokens, max_new_tokens=1024, do_sample=True, temperature=1.0, num_beams=5, no_repeat_ngram_size=2)
        output = output[:, tokens["input_ids"].shape[1]:]
        generated_text += tokenizer.batch_decode(output, skip_special_tokens=True)
    
print(generated_text[:5])

df['gemma_outputs'] = generated_text
df.to_csv('gemma7b.csv', index=False)
