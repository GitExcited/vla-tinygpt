"""Training loops, evaluation, and the two no-training diagnostics (weight-update edit and
logit common-part removal)."""
import math, time

import numpy as np
import torch
import torch.nn.functional as F

from .data import ATTRS, PAD


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def adamw(model, lr):
    # weight decay on every parameter, as in the reference setup
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def lm_loss(model, rows, device):
    x = torch.from_numpy(rows).to(device)
    logits = model(x[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), ignore_index=PAD)


def step(model, opt, rows, device):
    loss = lm_loss(model, rows, device)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


@torch.no_grad()
def answer_logits(model, ev, device, bs=500):
    """Logits that predict each answer token: [N_people, 6, V] (float32, on CPU)."""
    model.eval()
    out = []
    for i in range(0, len(ev.rows), bs):
        x = torch.from_numpy(ev.rows[i:i + bs]).to(device)
        logits = model(x[:, :-1])
        p = torch.from_numpy(ev.pos[i:i + bs]).to(device) - 1          # position that predicts it
        out.append(torch.gather(logits, 1, p[:, :, None].expand(-1, -1, logits.size(-1))).float().cpu())
    model.train()
    return torch.cat(out)


def score(z, ev, people=None):
    """Accuracy of full-vocabulary argmax, own-half top-1, and answer loss, from answer logits."""
    y = torch.from_numpy(ev.values).long()
    if people is not None:
        z, y = z[people], y[people]
    res = {"acc": (z.argmax(-1) == y).float().mean().item(),
           "loss": F.cross_entropy(z.reshape(-1, z.size(-1)), y.reshape(-1)).item()}
    if ev.own is not None:
        hits = []
        for k, a in enumerate(ATTRS):
            ids = torch.from_numpy(ev.own[a]).long()
            pick = ids[z[:, k][:, ids].argmax(-1)]
            hits.append((pick == y[:, k]).float())
        res["own_top1"] = torch.stack(hits, 1).mean().item()
    return res


def evaluate(model, evsets, device):
    """{name: metrics}. An eval entry is EvalSet or (EvalSet, person index array)."""
    out, cache = {}, {}
    for name, spec in evsets.items():
        ev, people = spec if isinstance(spec, tuple) else (spec, None)
        if id(ev) not in cache:
            cache[id(ev)] = answer_logits(model, ev, device)
        out[name] = score(cache[id(ev)], ev, people)
    return out


def pretrain(model, sampler, steps, batch, lr, final_lr, device, evsets, eval_every, tag):
    opt = adamw(model, lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: final_lr / lr + (1 - final_lr / lr) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    curve, t0 = [], time.time()
    for s in range(1, steps + 1):
        loss = step(model, opt, sampler.batch(batch), device)
        sched.step()
        if s % eval_every == 0 or s == steps:
            row = {"step": s, "train_loss": loss, **evaluate(model, evsets, device)}
            curve.append(row)
            eta = (time.time() - t0) / s * (steps - s) / 60
            log(f"  {tag} pretrain {s}/{steps} loss {loss:.3f} | "
                + " ".join(f"{k} {v['acc']:.2f}" for k, v in row.items() if isinstance(v, dict))
                + f" | ~{eta:.0f} min left")
    return curve


def train_constant(model, sampler, steps, batch, lr, device, evsets, eval_steps, ckpt_steps=()):
    """Fresh AdamW, constant LR (fine-tuning and relearning). Returns (curve, {step: CPU state})."""
    opt = adamw(model, lr)
    eval_steps, ckpts, curve = set(eval_steps), {}, []

    def snap(s):
        if s in ckpt_steps:
            ckpts[s] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if s in eval_steps:
            curve.append({"step": s, **evaluate(model, evsets, device)})

    snap(0)
    for s in range(1, steps + 1):
        step(model, opt, sampler.batch(batch), device)
        snap(s)
    return curve, ckpts


@torch.no_grad()
def update_svds(state0, state_t, names, device):
    """SVD of each block weight update dW = W(t) - W(0), computed once per checkpoint."""
    out = {}
    for n in names:
        U, S, Vh = torch.linalg.svd((state_t[n] - state0[n]).to(device).double(), full_matrices=False)
        out[n] = (U.cpu(), S.cpu(), Vh.cpu())
    return out


@torch.no_grad()
def edited_state(state_t, svds, rank, control, gen):
    """W(t) minus the top-`rank` part of each update dW = W(t) - W(0) (control=False), or minus
    a random rank-`rank` matrix with the same Frobenius norm as that top part (control=True)."""
    new = dict(state_t)
    for n, (U, S, Vh) in svds.items():
        top = (U[:, :rank] * S[:rank]) @ Vh[:rank]
        if control:
            a = torch.randn(top.shape[0], rank, generator=gen, dtype=torch.float64)
            b = torch.randn(rank, top.shape[1], generator=gen, dtype=torch.float64)
            r = a @ b
            top = r * (top.norm() / r.norm())
        new[n] = (state_t[n].double() - top).to(state_t[n].dtype)
    return new


def common_part_scores(z0, zt, ev):
    """Palit's logit split: c = mean over people of (z(t) - z(0)), one per attribute.
    Scores the trained logits, z(0) + c (common part only) and z(t) - c (common part removed)."""
    c = (zt - z0).mean(0, keepdim=True)
    return {"trained": score(zt, ev), "common_only": score(z0 + c, ev), "common_removed": score(zt - c, ev)}
