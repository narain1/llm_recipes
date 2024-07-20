from torch import nn
import os
import torch
import math
import inspect
import torch.nn.functional as F
# from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import Sampler
from data import DistributedDataLoader
# from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.nn.attention import SDPBackend
from model import Transformer
from functools import partial

import torch._inductor.config
import torch._dynamo.config
import wandb
from types import SimpleNamespace
from torch.optim.lr_scheduler import OneCycleLR
import argparse

hparams = SimpleNamespace(
        project_name = "test-loop-dist",
        lr = 1e-4,
        bs = 16,
        grad_accum_steps = 5,
        max_seq_length = 512,
        vocab_size = 32_000,
        m_dim = 768,
        grad_clip = 1.0,
        val_iterations = 200,
        max_steps = 50_000,
        d1 = "/scratch/npattab1/llm_data/tokens",
        wd = 0.1,
        device = "cuda",
        log_interval=10,
        dtype = 'bfloat16',
        compile = True,
        eval_interval = 500,
        # min_lr = 1e-7,
        # lr_decay = 0.02,
        warmup_iter = 500,
        # decat_ratio = 0.0,
)

def setup(rank, world_size, init_method="env://"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['OMP_NUM_THREADS'] = '16'
    dist.init_process_group('nccl', rank=rank, world_size=world_size, init_method=init_method)

def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument('--world_size', type=int, default=1, help='Total number of processes (GPUs) across all nodes')
    ddp_rank = int(os.environ.get('RANK', -1))
    if ddp_rank >= 0:
        ddp = True
        assert torch.cuda.is_available(), "cuda device not recognized"
        dist.init_process_group(backend="nccl")
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        print(f"Rank {ddp_rank}, Local Rank {ddp_local_rank}, Device {device}")
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = 0

    else:
        print("DDP rank not defined or invalid")
        ddp_local_rank = 0
        ddp_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = "cude" if torch.cuda.is_available() else "cpu"

    device_type = "cuda" if "cuda" else "cpu"
    hparams.dist_world_size = ddp_world_size

    if master_process:
        wandb.init(project=hparams.project_name,
               config = hparams
               )

    device_type = 'cuda' if 'cuda' in hparams.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[hparams.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else nullcontext()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # model = GPT(vocab_size, 768, 12, 12, 0.1, False, max_seq_length)
    model = Transformer.from_name("stories110M")
    model.train()
    model.to(hparams.device)
    if hparams.compile:
        torch._inductor.config.coordinate_descent_tuning = True
    #         torch._inductor.config.triton_unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        model = torch.compile(model)

    optimizer = model.configure_optimizers(hparams.wd, hparams.lr, betas=(0.9, 0.95),
                                               device_type=hparams.device) #, zero_stage=zero_stage)

    train_dl = DistributedDataLoader(hparams.d1, True, hparams.max_seq_length, hparams.bs, 1)
    val_dl = DistributedDataLoader(hparams.d1, False, hparams.max_seq_length, hparams.bs, 1)


    scheduler = OneCycleLR(
        optimizer,
        total_steps=hparams.max_steps,
        pct_start=hparams.warmup_iter/hparams.max_steps,
        max_lr = 5 * hparams.lr
    )
    
    step = 0
    model.train()


    while True:
        if step > 0 and step % hparams.eval_interval == 0:
           val_dl.reset()
           val_loss, val_acc = [], []
           model.eval()
           with torch.no_grad():
               for val_step in range(hparams.val_iterations):
                   x, y = val_dl.next_batch()
                   x, y = x.to(hparams.device), y.to(hparams.device)
                   logits, loss = model(x, y)
                   accuracy = (logits.view(-1, logits.size(-1)).argmax(-1) == y.view(-1)).float().mean()
                   val_acc.append(accuracy.item())
                   val_loss.append(loss.item())
           if master_process:
               wandb.log({'val_loss': np.mean(val_loss), 'val_acc': np.mean(val_acc)})
           model.train()
        optimizer.zero_grad(set_to_none=True)

        lossf = 0.0
        for microstep in range(hparams.grad_accum_steps):
           x, y = train_dl.next_batch()
           x, y = x.to(hparams.device), y.to(hparams.device)
           with ctx, torch.nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
               logits, loss = model(x, y)
               loss = loss / hparams.grad_accum_steps
               lossf += loss.detach()
           loss.backward()
        lossf = lossf.item()
        if master_process and step % hparams.log_interval == 0:
           wandb.log({"step": step, "loss": lossf, "lr": scheduler.get_last_lr()[0]})
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        torch.cuda.synchronize()

        if step > hparams.max_steps:
           break
        else:
            step += 1


    if master_process:
        torch.save(model.state_dict(), "model.pth")

    cleanup()
