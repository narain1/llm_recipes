from transformers import LlamaTokenizer, AutoModelForCausalLM
import transformers
import torch
import os

os.environ['TRANSFORMERS_CACHE'] = '/scratch/npattab1/hf_cache'
os.environ['HF_HOME'] = '/scratch/npattab1/hf_cache'
access_token = ""


model_id = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=access_token,
        cache_dir='/scratch/npattab1/llms/'
    )

tokenizer = LlamaTokenizer.from_pretrained(model_id, 
                                  token=access_token,
                                  cache_dir='/scratch/npattab1/llms/'
                                )



def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs


@torch.no_grad()
def greedy_decoding(
    model, prompt, seq_len, temperature=1.0, filter_thres=0.9):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None

    for _ in range(sample_num_times):
        logits, cache = model(out, cache=cache, return_cache=True)
        logits = logits[:, -1]

        logits = topk(logits, thres=filter_thres)
        sample = gumbel_sample(logits, temperature=temperature, dim=-1)
        out = torch.cat((out, sample[..., None]), dim=-1)
    return out[..., prompt_seq_len:]

inputs = tokenizer("narain was going through anna salai to reach beach will he reach the destination",
                   return_tensors='pt').to('cuda')


