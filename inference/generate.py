import itertools
import contextlib
import torch
from sentencepiece import SentencePieceProcessor

import torch._dynamo_config
import torch._inductor.config


tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left",
    add_eos_token=True,
    add_bos_token=True)


model = AutoModelForCausalLM.from_pretrained(model_path,
                                     torch_dtype=torch.float16,
                                    device_map="auto",
                                    trust_remote_code=True,
                                     )

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet supported")


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: int,
    callback= lambda x:x,
    **sampling_kwargs
)-> torch.Tensor:
    is_speculative = draft_model is not None
    device, dtype = prompt.device, prompt.dtype
    max_seq_length = 


if compile:
    global decode_one_token, prefill
    decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
    if compile_prefill:
        prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    start = -1 of compile else 0

    for i in range(start, num_samples):
        device_sync(device=device)

    if (i != num_samples - 1 or not profile) or (use_tp  and rank != 0):
        prof = contextlib.nullcontext()

    else:
        torch.profiler._utils._init_for_cuda_graphs()
        prof = torch.profiler.profile()
    with prof:
        y, metrics = generate(
            model,
            encoded,
            max_new_tokens,
            draft_model = draft_model,
            speculate_k=speculate_k,
            interactive=interactive,
            callback=callback,
            temperature=temperature,
            top_k=top_k
        )
        aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
    if i == -1:
        print(f"compilation time")
        continue
    if hasattr(prof, "export_chrome_trace"):
        if use_tp:
            prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
        else:
            prof.export_chrome_trace(f"{profile}.json")
    device_sync(device=device)
    t = time.perf_counter() - t0

    if not interactive:
        print(tokenizer.decode(y.tolist()))
    else:
        print()

    tokens
