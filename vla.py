"""VLA with a tiny GPT: does a model that forgot a domain relearn it faster?

Arms (all start from the same random weights; stages 2 and 3 use the same data):
  A  stage 1 on TinyStories  -> stage 2 on code -> stage 3 on stories
  B  (no stage 1)            -> stage 2 on code -> stage 3 on stories
  C  stage 1 on WikiText-103 -> stage 2 on code -> stage 3 on stories
  S  (no stage 1 or 2)                          -> stage 3 on stories   (learning from scratch)

Stage 3 runs twice per arm, both copies starting from the same weights:
  seen  = the first `relearn_tokens` of A's stage-1 data (only A has read it)
  fresh = TinyStories no arm has trained on
Curves are scored on held-out TinyStories (pre_val). A "probe" also scores the seen and
fresh piles directly at a few points: if A scores better on the seen pile than on the fresh
one when stage 3 starts, specific text survived stage 2 (memorization), not just the language.

End-of-stage-2 models are saved in --ckpt-dir, so stage-3 variants (e.g. --set
relearn_steps=120) skip stages 1-2. Results are one JSON per (seed, arm) in
results/<tag>/; a finished (seed, arm) is skipped on restart.

usage:
  python vla.py --seeds 0 1 2 --mem-frac 0.33                      # real run (GPU)
  python vla.py --seeds 0 --tag onepass --set relearn_steps=120    # stage-3 variant
  python vla.py --smoke --seeds 0                                  # tiny synthetic run, CPU ok
then:  python plots.py results/main
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS

import argparse, copy, hashlib, json, random, time
from dataclasses import dataclass, asdict, fields, replace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    seed: int = 0
    smoke: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # model
    block_size: int = 256
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    # optimization (a fresh AdamW at the start of every stage, constant LR)
    batch_size: int = 32
    lr: float = 6e-4
    orth_lr: Optional[float] = None      # stage-2 LR; None = lr
    relearn_lr: Optional[float] = None   # stage-3 LR; None = lr
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # stage lengths
    pretrain_steps: int = 5000
    orth_steps: int = 5000
    relearn_steps: int = 500
    # evaluation
    eval_every: int = 100          # stages 1-2
    relearn_eval_every: int = 25   # stage 3
    target_tol: float = 0.05       # "recovered" = within 5% of A's end-of-stage-1 loss
    # data sizes (tokens)
    relearn_tokens: int = 1_000_000   # ~2.4% of the ~41M stage-1 tokens
    val_tokens: int = 300_000
    data_dir: str = "data_cache"

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * (self.block_size + 1)


SMOKE_OVERRIDES = dict(
    smoke=True, block_size=32, d_model=64, n_layer=2, n_head=4,
    batch_size=16, lr=3e-3, weight_decay=0.0,
    pretrain_steps=1500, orth_steps=1500, relearn_steps=300,
    eval_every=50, relearn_eval_every=10,
    relearn_tokens=16 * 33 * 20, val_tokens=20_000,
)

# fields that change what stages 1-2 produce; a saved checkpoint is reused only if all match
STAGE12_FIELDS = ("seed", "smoke", "block_size", "d_model", "n_layer", "n_head", "batch_size",
                  "lr", "orth_lr", "weight_decay", "grad_clip", "pretrain_steps", "orth_steps")

ARMS = {"A": "pre_A", "B": None, "C": "pre_C", "S": None}   # arm -> stage-1 dataset
SUBSETS = ["seen", "fresh"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------- data

def make_markov_language(vocab: int, rng: np.random.Generator, branching: int = 4):
    """A synthetic 'language': each token can be followed by only a few tokens."""
    nxt = rng.integers(0, vocab, size=(vocab, branching))
    probs = rng.dirichlet(np.ones(branching) * 0.5, size=vocab)   # peaky rows
    return nxt, probs


def sample_markov(lang, n_tokens: int, rng: np.random.Generator) -> np.ndarray:
    nxt, probs = lang
    cum = np.cumsum(probs, axis=1)
    u = rng.random(n_tokens)
    out = np.empty(n_tokens, dtype=np.uint16)
    t = int(rng.integers(0, nxt.shape[0]))
    for i in range(n_tokens):
        out[i] = t
        j = min(int(np.searchsorted(cum[t], u[i])), nxt.shape[1] - 1)
        t = int(nxt[t, j])
    return out


def load_smoke_data(cfg: Config):
    """Three unrelated synthetic languages standing in for TinyStories / WikiText / code."""
    vocab = 256
    rng = np.random.default_rng(1234)
    L_pre, L_other, L_orth = (make_markov_language(vocab, rng) for _ in range(3))
    n_pre = cfg.pretrain_steps * cfg.tokens_per_step
    n_orth = cfg.orth_steps * cfg.tokens_per_step
    pre_all = sample_markov(L_pre, n_pre + cfg.relearn_tokens, rng)
    data = {
        "pre_A": pre_all[:n_pre],
        "pre_C": sample_markov(L_other, n_pre, rng),
        "orth": sample_markov(L_orth, n_orth, rng),
        "orth_val": sample_markov(L_orth, cfg.val_tokens, rng),
        "pre_val": sample_markov(L_pre, cfg.val_tokens, rng),
        "relearn_seen": pre_all[:cfg.relearn_tokens],
        "relearn_fresh": pre_all[n_pre:n_pre + cfg.relearn_tokens],
    }
    return data, vocab


def _stream_tokens(enc, n_tokens, path, field, name=None, split="train"):
    """Tokenize documents from a streamed HF dataset until n_tokens are collected."""
    from datasets import load_dataset
    ds = load_dataset(path, name, split=split, streaming=True)
    pieces, total = [], 0
    for ex in ds:
        text = ex[field]
        if not text or not text.strip():
            continue
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)                 # document separator
        pieces.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        if total >= n_tokens:
            break
    if total < n_tokens:
        raise RuntimeError(f"{path}: only {total} tokens available, wanted {n_tokens}")
    return np.concatenate(pieces)[:n_tokens]


def load_real_data(cfg: Config):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")           # vocab 50257 < 65536, so uint16 is enough
    n_pre = cfg.pretrain_steps * cfg.tokens_per_step
    n_orth = cfg.orth_steps * cfg.tokens_per_step
    os.makedirs(cfg.data_dir, exist_ok=True)
    cache = os.path.join(cfg.data_dir, f"real_{n_pre}_{n_orth}_{cfg.relearn_tokens}_{cfg.val_tokens}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache)), enc.n_vocab

    log("downloading + tokenizing data (first run only)")
    pre_all = _stream_tokens(enc, n_pre + cfg.relearn_tokens, "roneneldan/TinyStories", "text")
    pre_val = _stream_tokens(enc, cfg.val_tokens, "roneneldan/TinyStories", "text", split="validation")
    pre_C = _stream_tokens(enc, n_pre, "Salesforce/wikitext", "text", name="wikitext-103-raw-v1")
    orth_all = _stream_tokens(enc, n_orth + cfg.val_tokens, "codeparrot/codeparrot-clean-valid", "content")
    data = {
        "pre_A": pre_all[:n_pre],
        "pre_C": pre_C,
        "orth": orth_all[:n_orth],
        "orth_val": orth_all[n_orth:],
        "pre_val": pre_val,
        "relearn_seen": pre_all[:cfg.relearn_tokens],
        "relearn_fresh": pre_all[n_pre:],
    }
    np.savez(cache, **data)
    return data, enc.n_vocab


def to_chunks(tokens, block_size):
    row_len = block_size + 1
    num_rows = len(tokens) // row_len
    if num_rows == 0:
        raise ValueError(f"need at least {row_len} tokens, got {len(tokens)}")
    return torch.from_numpy(tokens[:num_rows * row_len].astype(np.int64)).view(num_rows, row_len)


class BatchStream:
    def __init__(self, chunks, batch_size, seed, device):
        if chunks.shape[0] < batch_size:
            raise ValueError(f"{chunks.shape[0]} rows can't fill a batch of {batch_size}")
        self.chunks, self.batch_size, self.device = chunks, batch_size, device
        self.gen = torch.Generator().manual_seed(seed)   # private RNG, never the global one
        self._new_epoch()

    def _new_epoch(self):
        self.order = torch.randperm(self.chunks.shape[0], generator=self.gen)
        self.pos = 0

    def next(self):
        if self.pos + self.batch_size > len(self.order):
            self._new_epoch()
        idx = self.order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        batch = self.chunks[idx]
        return batch[:, :-1].to(self.device), batch[:, 1:].to(self.device)


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config, vocab: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.head.weight = self.tok.weight        # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ---------------------------------------------------------------- training

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def eval_loss(model, chunks, cfg: Config) -> float:
    """Mean next-token loss over ALL rows of a fixed held-out set, in fixed order."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, chunks.shape[0], cfg.batch_size):
        batch = chunks[i:i + cfg.batch_size].to(cfg.device)
        loss = model(batch[:, :-1], batch[:, 1:])[1]
        total += loss.item() * batch.shape[0]
        n += batch.shape[0]
    model.train()
    return total / n


def train_stage(model, stream, steps, lr, cfg, eval_sets, eval_every, stage, step_offset=0):
    """Train for `steps` steps; return a log of evals (including step 0, before any update)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    model.train()
    rows = []

    def record(step, train_loss):
        row = {"stage": stage, "step": step, "global_step": step_offset + step, "train_loss": train_loss}
        row.update({name: eval_loss(model, ch, cfg) for name, ch in eval_sets.items()})
        rows.append(row)

    record(0, None)
    for s in range(1, steps + 1):
        x, y = stream.next()
        loss = model(x, y)[1]        # [1] only: holding the logits through backward wastes ~1.5 GB
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if s % 1000 == 0:
            log(f"    {stage} step {s}/{steps}")
        if s % eval_every == 0 or s == steps:
            record(s, loss.item())
    return rows


def probe(model, chunks, cfg: Config) -> dict:
    """Loss on the held-out story test and directly on the two stage-3 piles."""
    return {name: eval_loss(model, chunks[key], cfg)
            for name, key in (("pre_val", "pre_val"), ("seen_pile", "relearn_seen"),
                              ("fresh_pile", "relearn_fresh"))}


def stage12_key(cfg: Config) -> str:
    blob = json.dumps({f: getattr(cfg, f) for f in STAGE12_FIELDS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def run_seed(cfg: Config, chunks: dict, vocab: int, arms, out_dir: str, ckpt_dir: str):
    set_seed(cfg.seed)
    init_state = copy.deepcopy(TinyGPT(cfg, vocab).state_dict())   # same init for every arm
    eval_sets = {"pre_val": chunks["pre_val"], "orth_val": chunks["orth_val"]}
    orth_lr = cfg.orth_lr or cfg.lr
    relearn_lr = cfg.relearn_lr or cfg.lr

    def model_from(state):
        m = TinyGPT(cfg, vocab).to(cfg.device)
        m.load_state_dict(state)
        return m

    for arm in arms:
        stage1 = ARMS[arm]
        out = os.path.join(out_dir, f"seed{cfg.seed}_arm{arm}.json")
        if os.path.exists(out):
            log(f"seed {cfg.seed} arm {arm}: already done ({out})")
            continue
        t0 = time.time()
        ckpt = os.path.join(ckpt_dir, f"seed{cfg.seed}_arm{arm}_{stage12_key(cfg)}.pt")

        if arm == "S":                                   # no stages 1-2
            after12, stages12, probes = init_state, [], {}
        elif os.path.exists(ckpt):
            saved = torch.load(ckpt, map_location="cpu")
            after12, stages12, probes = saved["model"], saved["stages12"], saved["probes"]
            log(f"seed {cfg.seed} arm {arm}: stages 1-2 loaded from {ckpt}")
        else:
            model = model_from(init_state)
            stages12, probes, g = [], {}, 0
            if stage1 is not None:
                # same stream seed for A and C: same shuffle order, different data
                s1 = BatchStream(chunks[stage1], cfg.batch_size, cfg.seed + 1, cfg.device)
                stages12 += train_stage(model, s1, cfg.pretrain_steps, cfg.lr, cfg, eval_sets,
                                        cfg.eval_every, "1_pretrain", g)
                g += cfg.pretrain_steps
                probes["after_stage1"] = probe(model, chunks, cfg)
            s2 = BatchStream(chunks["orth"], cfg.batch_size, cfg.seed + 2, cfg.device)
            stages12 += train_stage(model, s2, cfg.orth_steps, orth_lr, cfg, eval_sets,
                                    cfg.eval_every, "2_orth", g)
            after12 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"model": after12, "stages12": stages12, "probes": probes,
                        "config": asdict(cfg)}, ckpt)
            del model

        offset = 0 if arm == "S" else (cfg.pretrain_steps if stage1 else 0) + cfg.orth_steps
        relearn = {}
        for sub in SUBSETS:
            m = model_from(after12)                      # both piles branch from the same weights
            if sub == SUBSETS[0]:
                probes["stage3_start"] = probe(m, chunks, cfg)
            s3 = BatchStream(chunks[f"relearn_{sub}"], cfg.batch_size, cfg.seed + 3, cfg.device)
            relearn[sub] = train_stage(m, s3, cfg.relearn_steps, relearn_lr, cfg,
                                       {"pre_val": chunks["pre_val"]}, cfg.relearn_eval_every,
                                       f"3_relearn_{sub}", offset)
            probes[f"stage3_end_{sub}"] = probe(m, chunks, cfg)
            del m

        with open(out, "w") as f:
            json.dump({"arm": arm, "seed": cfg.seed, "config": asdict(cfg),
                       "stages12": stages12, "probes": probes, "relearn": relearn}, f, indent=1)
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        start = probes["stage3_start"]
        log(f"seed {cfg.seed} arm {arm} done in {time.time() - t0:.0f}s | stage-3 start: "
            f"story test {start['pre_val']:.3f}, seen pile {start['seen_pile']:.3f}, "
            f"fresh pile {start['fresh_pile']:.3f} | end (fresh): {relearn['fresh'][-1]['pre_val']:.3f}")


def parse_set(items):
    """--set key=value overrides, typed by the Config field."""
    types = {f.name: f.type for f in fields(Config)}
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown config field: {k}")
        t = types[k]
        if v.lower() == "none":
            out[k] = None
        elif t is bool:
            out[k] = v.lower() in ("1", "true", "yes")
        elif t is int:
            out[k] = int(v)
        elif "float" in str(t):          # float or Optional[float]
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--tag", default="main", help="results go to <results-dir>/<tag>/")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic data and model")
    p.add_argument("--mem-frac", type=float, default=None,
                   help="cap this process's share of GPU memory (0.33 = 32 GB on the lab server)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override Config fields")
    args = p.parse_args()

    cfg = Config()
    if args.smoke:
        cfg = replace(cfg, **SMOKE_OVERRIDES)
    cfg = replace(cfg, **parse_set(args.set))
    if cfg.device == "cuda" and args.mem_frac:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)

    log(f"device {cfg.device}" + (f" ({torch.cuda.get_device_name(0)})" if cfg.device == "cuda" else ""))
    data, vocab = load_smoke_data(cfg) if cfg.smoke else load_real_data(cfg)
    chunks = {k: to_chunks(v, cfg.block_size) for k, v in data.items()}
    out_dir = os.path.join(args.results_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    for seed in args.seeds:
        log(f"=== seed {seed} ===")
        run_seed(replace(cfg, seed=seed), chunks, vocab, args.arms, out_dir, args.ckpt_dir)
    log(f"all done -> {out_dir}   (plot with: python plots.py {out_dir})")


if __name__ == "__main__":
    main()
