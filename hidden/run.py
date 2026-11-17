"""Gone or hidden? Palit-style biographies, plus a never-had-it twin and an untaught-half test.

For each seed:
  1. Pretrain two models from the same initial weights:
       main: A + ballast        twin: A_twin + ballast   (A_twin = different people, same kind)
  2. Fine-tune each on new people, no replay, under two conditions:
       disjoint (Palit's B': new answers in ballast's half) and all (Palit's C: both halves).
     Score every 10 steps; save the weights at CKPT steps along the curve.
  3. From every saved checkpoint, for both models:
     a. Relearn: train on the first half of A only ("taught"), score both halves.
        The second half ("untaught") is never shown, so its score can only rise if its facts
        were still stored. The twin never saw A at all: its untaught score is the baseline.
     b. Edit (no training): remove the top 1, 2, 4, ... parts of each block weight update,
        and, as a control, random parts of the same size; score old and new facts.
     c. Logit common-part removal (Palit's Fig. 4 test) on the model's own old set.
Writes results/<out>/seed<N>_<cond>.json.  Resumes: finished files are skipped,
pretrained models are cached in --ckpt-dir.

usage:  python -m hidden.run --seeds 0 1 2            (GPU)
        python -m hidden.run --smoke --seeds 0        (tiny, CPU, checks the pipeline)
"""
import argparse, json, os, time
from dataclasses import dataclass, asdict, field, replace

import numpy as np
import torch

from .data import EvalSet, Sampler, build_world
from .model import AttnOnlyLM
from .train import (answer_logits, common_part_scores, edited_state, evaluate, log, pretrain,
                    train_constant, update_svds)


@dataclass
class Cfg:
    n_a: int = 2000
    n_ballast: int = 2000
    n_new: int = 500
    pool_size: int = 130
    year_pool: int = 220
    d: int = 512
    n_layer: int = 8
    n_head: int = 8
    batch: int = 256
    pre_steps: int = 16000
    pre_lr: float = 5e-4
    pre_final_lr: float = 1e-7
    pre_eval_every: int = 1000
    ft_steps: int = 1200
    ft_lr: float = 3.75e-5
    ft_eval_every: int = 10
    ckpt_steps: list = field(default_factory=lambda: [0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200])
    relearn_steps: int = 400
    relearn_lr: float = 3.75e-5
    relearn_eval: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50,
                                                        75, 100, 150, 200, 300, 400])
    edit_ranks: list = field(default_factory=lambda: [1, 2, 4, 8, 16])
    conds: list = field(default_factory=lambda: ["disjoint", "all"])


SMOKE = dict(n_a=200, n_ballast=200, n_new=50, pool_size=30, year_pool=40, d=64, n_layer=2, n_head=4,
             batch=32, pre_steps=300, pre_eval_every=100, ft_steps=60, ckpt_steps=[0, 20, 60],
             relearn_steps=20, relearn_eval=[0, 1, 2, 5, 10, 20], edit_ranks=[1, 2])


def new_model(cfg, world, device):
    return AttnOnlyLM(len(world.vocab), cfg.d, cfg.n_layer, cfg.n_head).to(device)


def load(model, state):
    model.load_state_dict(state)
    return model


def run_seed(cfg, seed, out_dir, ckpt_dir, device):
    world = build_world(seed, cfg.n_a, cfg.n_ballast, cfg.n_new, cfg.pool_size, cfg.year_pool)
    half = cfg.n_a // 2
    taught, untaught = np.arange(half), np.arange(half, cfg.n_a)
    ev = {"A": EvalSet(world, "A", range(cfg.n_a), 200 + seed),
          "A_twin": EvalSet(world, "A_twin", range(cfg.n_a), 201 + seed),
          "ballast": EvalSet(world, "ballast", range(cfg.n_ballast), 202 + seed)}
    for c in cfg.conds:
        ev[f"new_{c}"] = EvalSet(world, "new", range(cfg.n_new), 203 + seed, cond=c)
    old_set = {"main": "A", "twin": "A_twin"}

    # 1. pretraining (cached)
    torch.manual_seed(seed)
    init = {k: v.clone() for k, v in new_model(cfg, world, "cpu").state_dict().items()}
    pre_states, pre_curves = {}, {}
    for variant, sets in (("main", ["A", "ballast"]), ("twin", ["A_twin", "ballast"])):
        path = os.path.join(ckpt_dir, f"pre_seed{seed}_{variant}_{'smoke' if cfg.d < 128 else 'full'}.pt")
        if os.path.exists(path):
            saved = torch.load(path, map_location="cpu")
            pre_states[variant], pre_curves[variant] = saved["state"], saved["curve"]
            log(f"seed {seed} {variant}: pretrained model loaded")
            continue
        m = load(new_model(cfg, world, device), init)
        evs = {"A": ev["A"], "A_twin": ev["A_twin"], "ballast": ev["ballast"]}
        pre_curves[variant] = pretrain(m, Sampler(world, sets, 43 + seed), cfg.pre_steps, cfg.batch,
                                       cfg.pre_lr, cfg.pre_final_lr, device, evs, cfg.pre_eval_every,
                                       f"seed {seed} {variant}")
        pre_states[variant] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"state": pre_states[variant], "curve": pre_curves[variant]}, path)

    for cond in cfg.conds:
        out = os.path.join(out_dir, f"seed{seed}_{cond}.json")
        if os.path.exists(out):
            log(f"seed {seed} {cond}: done already")
            continue
        t0 = time.time()
        result = {"seed": seed, "cond": cond, "config": asdict(cfg), "pretrain": pre_curves}
        ft_evs = {"A_taught": (ev["A"], taught), "A_untaught": (ev["A"], untaught), "A": ev["A"],
                  "A_twin": ev["A_twin"], "ballast": ev["ballast"], "new": ev[f"new_{cond}"]}
        for variant in ("main", "twin"):
            # 2. fine-tune on new facts, no replay
            m = load(new_model(cfg, world, device), pre_states[variant])
            ft_steps = list(range(0, cfg.ft_steps + 1, cfg.ft_eval_every))
            curve, ckpts = train_constant(m, Sampler(world, ["new"], 47 + seed, cond=cond), cfg.ft_steps,
                                          cfg.batch, cfg.ft_lr, device, ft_evs, ft_steps, set(cfg.ckpt_steps))
            log(f"seed {seed} {cond} {variant}: fine-tuned ({time.time() - t0:.0f}s)")
            names = m.block_matrices()
            old = ev[old_set[variant]]
            z0 = answer_logits(load(m, ckpts[0]), old, device)
            per_ckpt = {}
            for t in cfg.ckpt_steps:
                rec = {}
                # 3a. relearn the taught half of A, score both halves
                r = load(new_model(cfg, world, device), ckpts[t])
                rec["relearn"], _ = train_constant(
                    r, Sampler(world, {"A": taught}, 500 + seed + t), cfg.relearn_steps, cfg.batch,
                    cfg.relearn_lr, device,
                    {"A_taught": (ev["A"], taught), "A_untaught": (ev["A"], untaught), "new": ev[f"new_{cond}"]},
                    cfg.relearn_eval)
                # 3b. edits of the weight update (no training)
                gen = torch.Generator().manual_seed(seed * 1000 + t)
                edit_evs = {"old": old, "ballast": ev["ballast"], "new": ev[f"new_{cond}"]}
                rec["edit"] = {"none": evaluate(load(m, ckpts[t]), edit_evs, device)}
                svds = update_svds(ckpts[0], ckpts[t], names, device)
                for k in cfg.edit_ranks:
                    for control in (False, True):
                        st = edited_state(ckpts[t], svds, k, control, gen)
                        rec["edit"][f"{'random' if control else 'top'}{k}"] = evaluate(load(m, st), edit_evs, device)
                # 3c. logit common-part removal on the model's own old set
                zt = answer_logits(load(m, ckpts[t]), old, device)
                rec["common_part"] = common_part_scores(z0, zt, old)
                per_ckpt[str(t)] = rec
            result[variant] = {"finetune": curve, "checkpoints": per_ckpt}
            log(f"seed {seed} {cond} {variant}: relearn + edits done ({time.time() - t0:.0f}s)")
        with open(out, "w") as f:
            json.dump(result, f)
        log(f"seed {seed} {cond}: wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/hidden")
    p.add_argument("--ckpt-dir", default="checkpoints/hidden")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override Cfg fields, e.g. --set d=256 pre_steps=8000 (lists: 0,50,100)")
    args = p.parse_args()
    cfg = replace(Cfg(), **SMOKE) if args.smoke else Cfg()
    over = {}
    for item in args.set:
        k, v = item.split("=", 1)
        cur = getattr(cfg, k)
        over[k] = [int(x) for x in v.split(",")] if isinstance(cur, list) and k != "conds" else             v.split(",") if k == "conds" else type(cur)(v)
    cfg = replace(cfg, **over)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    os.makedirs(args.out, exist_ok=True)
    for seed in args.seeds:
        run_seed(cfg, seed, args.out, args.ckpt_dir, device)
    log(f"all done -> {args.out}   (summarize with: python -m hidden.report {args.out})")


if __name__ == "__main__":
    main()
