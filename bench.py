"""Time 200 training steps of the real-config model, to estimate a full run on this GPU.

usage: CUDA_VISIBLE_DEVICES=1 python bench.py [--mem-frac 0.33]
"""
import argparse, time

import torch

from vla import Config, TinyGPT, set_seed

p = argparse.ArgumentParser()
p.add_argument("--mem-frac", type=float, default=0.33)
args = p.parse_args()
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
set_seed(0)

cfg, V = Config(), 50257
m = TinyGPT(cfg, V).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def step():
    x = torch.randint(0, V, (cfg.batch_size, cfg.block_size + 1), device="cuda")   # speed doesn't depend on the data
    loss = m(x[:, :-1], x[:, 1:])[1]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.grad_clip)
    opt.step()


print(torch.cuda.get_device_name(0))
for _ in range(20):                       # warm-up, not timed
    step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(200):
    step()
torch.cuda.synchronize()
per1k = (time.time() - t0) / 200 * 1000 / 60
# one seed, all 4 arms: A and C 5000+5000+2x500, B 5000+2x500, S 2x500 = 30k steps, plus evals (~+5%)
print(f"{per1k:.2f} min per 1000 steps   (RTX 3080 desktop: ~3.0)")
print(f"one seed (4 arms) ~ {per1k * 31.5:.0f} min   (RTX 3080 desktop: ~95)")
print(f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
