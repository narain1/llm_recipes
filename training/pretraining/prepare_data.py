import sentencepiece as spm
import os
from tqdm import tqdm
import numpy as np

tokenizer = spm.SentencePieceProcessor()
tokenizer.load("out/tamil_spm.model")

data_dir = "data/chunks"
data_fs = os.listdir(data_dir)
token_dir = "data/tokens"

os.makedirs(token_dir, exist_ok=True)

for file in tqdm(data_fs):
    with open(os.path.join(data_dir, file), "r") as f_read:
        o = f_read.read()
    all_tokens = []
    for line in o.split('\n'):
        all_tokens.extend(tokenizer.encode(line))
    all_tokens = np.array(all_tokens, dtype=np.int16)
    with open(os.path.join(token_dir, file.split('.')[0] + '.bin'), "wb") as f_write:
        f_write.write(all_tokens.tobytes())


