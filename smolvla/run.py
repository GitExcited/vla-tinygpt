"""The VLA version of the TinyGPT experiment: SmolVLA on LIBERO.

LIBERO task suites play the role of the text domains:
  A  stage 1 LIBERO-Object (half the demos) -> stage 2 LIBERO-Goal -> stage 3 Object (other half)
  C  stage 1 LIBERO-Spatial (same # demos)  -> stage 2 LIBERO-Goal -> stage 3 Object   (the twin)
  B  (no stage 1)                           -> stage 2 LIBERO-Goal -> stage 3 Object
  S  (no stage 1 or 2)                                             -> stage 3 Object
Every arm starts from the same pretrained SmolVLA; every stage starts a fresh optimizer from the
previous stage's weights (lerobot-train). Stage 3 uses demos that no arm trained on before.

Scores
  - stage-3 training loss, logged every few steps: the first pass over unseen demos, the
    equivalent of TinyGPT's story-test loss;
  - success rate on all 10 LIBERO-Object tasks in the simulator (lerobot-eval), after stage 1
    (A's peak), and at stage-3 steps EVAL_STEPS (step 0 = right after forgetting).

Before the first real run, a smoke test picks how to start from the pretrained model: the
robot-pretrained smolvla_base (two ways of mapping its camera names), and if neither works with
the LIBERO data, SmolVLA initialized from its pretrained vision-language model. The choice is saved
in <out>/init.json and reported in every result file.

Resumable: finished stages, evaluations and arms are skipped. Disk: stage checkpoints are deleted
once nothing needs them.

usage (inside the env_vla environment, MUJOCO_GL=egl):
  python -m smolvla.run --smoke                  # ~10-20 min: checks everything end to end
  python -m smolvla.run --seeds 0 1 2            # the real run, seeds one after another
"""
import argparse, glob, importlib.util, json, os, random, re, shutil, subprocess, sys, time
from dataclasses import asdict, dataclass, field, replace

from .tasks import TASK_TO_SUITE, norm

DATASET = os.environ.get("VLA_DATASET", "lerobot/libero")
# VLA_FULL=1: train the whole model (vision encoder + language model + action expert), as the VLA
# paper does for Pi0. Default (unset): smolvla_base's own setting, only the action expert trains.
FULL = os.environ.get("VLA_FULL", "") not in ("", "0")
FULL_ARGS = ["--policy.train_expert_only=false", "--policy.freeze_vision_encoder=false"] if FULL else []
ARMS = {"A": ["object_pre", "goal", "object_fresh"],
        "C": ["spatial", "goal", "object_fresh"],
        "B": ["goal", "object_fresh"],
        "S": ["object_fresh"]}
RENAME = '{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
INIT_VARIANTS = [
    {"name": "smolvla_base+rename", "base": ["--policy.path=lerobot/smolvla_base"],
     "extra": [f"--rename_map={RENAME}"]},
    {"name": "smolvla_base", "base": ["--policy.path=lerobot/smolvla_base"], "extra": []},
    {"name": "smolvla_from_vlm", "base": ["--policy.type=smolvla", "--policy.load_vlm_weights=true"], "extra": []},
]


@dataclass
class Cfg:
    stage_steps: int = 8000          # stages 1 and 2
    relearn_steps: int = 3000        # stage 3
    batch_size: int = 32
    num_workers: int = 8
    log_freq: int = 50               # stages 1-2
    relearn_log_freq: int = 5        # stage 3
    relearn_save_freq: int = 500
    eval_steps: list = field(default_factory=lambda: [0, 500, 1500, 3000])
    eval_episodes: int = 10          # per task, 10 tasks
    eval_batch: int = 5              # simulators in parallel
    eval_task_ids: str = ""          # "" = all 10 tasks


SMOKE = dict(stage_steps=20, relearn_steps=40, batch_size=4, num_workers=2, log_freq=5, relearn_log_freq=5,
             relearn_save_freq=20, eval_steps=[0, 40], eval_episodes=1, eval_batch=1, eval_task_ids="[0]")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, log_path):
    """Run a command, stream its output into log_path, return (exit code, output)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        lines = []
        for line in p.stdout:
            f.write(line)
            lines.append(line)
        p.wait()
    return p.returncode, "".join(lines)


def tail(text, n=25):
    return "\n".join("      | " + l for l in text.strip().splitlines()[-n:])


# ---------------------------------------------------------------- data

def episode_sets(out):
    """Sort the dataset's episodes into suites from its metadata files; split LIBERO-Object
    per task into two halves (pre / fresh); take as many Spatial demos as Object-pre has."""
    path = os.path.join(out, "episodes.json")
    if os.path.exists(path):
        return json.load(open(path))
    import pandas as pd
    from huggingface_hub import snapshot_download
    root = snapshot_download(DATASET, repo_type="dataset", allow_patterns=["meta/*", "meta/**"])
    ep_task = {}
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    if files and "tasks" in pd.read_parquet(files[0]).columns:   # v3 with task names per episode
        df = pd.concat([pd.read_parquet(f) for f in files])
        for _, r in df.iterrows():
            t = r["tasks"]
            ep_task[int(r["episode_index"])] = t if isinstance(t, str) else list(t)[0]
    elif files:                                                   # v3: task index lives in the frame data
        tasks = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))
        name = {int(i): t for t, i in zip(tasks.index, tasks["task_index"])}
        root = snapshot_download(DATASET, repo_type="dataset", allow_patterns=["data/*", "data/**"])
        for f in sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True)):
            d = pd.read_parquet(f, columns=["episode_index", "task_index"]).drop_duplicates("episode_index")
            for e, t in zip(d["episode_index"], d["task_index"]):
                ep_task[int(e)] = name[int(t)]
    elif os.path.exists(os.path.join(root, "meta", "episodes.jsonl")):   # v2.x
        for line in open(os.path.join(root, "meta", "episodes.jsonl")):
            r = json.loads(line)
            ep_task[int(r["episode_index"])] = r["tasks"][0]
    else:
        raise SystemExit(f"no episode metadata found under {root}/meta")
    by_suite, unknown = {}, set()
    for ep, t in sorted(ep_task.items()):
        s = TASK_TO_SUITE.get(norm(t))
        if s is None:
            unknown.add(t)
            continue
        by_suite.setdefault(s, {}).setdefault(norm(t), []).append(ep)
    for s in ("libero_object", "libero_goal", "libero_spatial"):
        n_tasks = len(by_suite.get(s, {}))
        if n_tasks != 10:
            raise SystemExit(f"{s}: matched {n_tasks}/10 tasks. Unmatched task strings: {sorted(unknown)[:50]}")
    pre, fresh = [], []
    for t, eps in sorted(by_suite["libero_object"].items()):
        eps = sorted(eps)
        pre += eps[0::2]                                          # alternate, so both halves span the task
        fresh += eps[1::2]
    spatial = sorted(e for eps in by_suite["libero_spatial"].values() for e in eps)
    spatial = sorted(random.Random(0).sample(spatial, min(len(pre), len(spatial))))
    goal = sorted(e for eps in by_suite["libero_goal"].values() for e in eps)
    sets = {"object_pre": sorted(pre), "object_fresh": sorted(fresh), "spatial": spatial, "goal": goal}
    os.makedirs(out, exist_ok=True)
    json.dump(sets, open(path, "w"))
    log("episodes: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
    return sets


def video_backend():
    return "torchcodec" if importlib.util.find_spec("torchcodec") else "pyav"


def dataset_root():
    base = os.environ.get("HF_LEROBOT_HOME") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "lerobot")
    return os.path.join(base, DATASET)


def ensure_full_dataset():
    """Download the WHOLE dataset once. lerobot-train with --dataset.episodes only fetches the
    files holding those episodes, and a later stage asking for other episodes then finds the
    folder present but its files missing ("Instruction 'train' corresponds to no data")."""
    from huggingface_hub import snapshot_download
    root = dataset_root()
    log(f"dataset: making sure all of {DATASET} is in {root} (1.9 GB, first time only)")
    snapshot_download(DATASET, repo_type="dataset", local_dir=root)
    return root


# ---------------------------------------------------------------- train / eval

def ckpt(stage_dir, which="last"):
    """Checkpoint folder of a stage: 'last', or the one saved at step `which` (any zero padding)."""
    base = os.path.join(stage_dir, "checkpoints")
    if which != "last" and os.path.isdir(base):
        for name in os.listdir(base):
            if name.isdigit() and int(name) == int(which):
                return os.path.join(base, name, "pretrained_model")
    return os.path.join(base, "last" if which == "last" else f"{int(which):06d}", "pretrained_model")


def train(cfg, init_args, extra, episodes, steps, stage_dir, seed, save_freq, log_freq):
    """One stage of lerobot-train. Returns the logged training losses (one per log_freq steps)."""
    done = os.path.join(stage_dir + ".done.json")
    if os.path.exists(done):
        return json.load(open(done))["loss"]
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)                                  # unfinished: start the stage again
    cmd = ["lerobot-train", *init_args, "--policy.push_to_hub=false",
           f"--dataset.repo_id={DATASET}", f"--dataset.root={dataset_root()}",
           f"--dataset.episodes=[{','.join(map(str, episodes))}]",
           f"--dataset.video_backend={video_backend()}", f"--batch_size={cfg.batch_size}",
           f"--num_workers={cfg.num_workers}", f"--steps={steps}", f"--save_freq={save_freq}",
           f"--log_freq={log_freq}", "--eval_freq=0", "--wandb.enable=false", f"--seed={seed}",
           f"--output_dir={stage_dir}", *extra, *FULL_ARGS]
    t0 = time.time()
    rc, out = run(cmd, stage_dir + ".log")
    if rc != 0 or not os.path.exists(ckpt(stage_dir)):
        raise RuntimeError(f"lerobot-train failed (exit {rc}), log: {stage_dir}.log\n{tail(out)}")
    loss = [float(m.group(1)) for m in re.finditer(r"step:\S+.*?\bloss:([0-9.eE+-]+)", out)]
    json.dump({"loss": loss, "log_freq": log_freq, "steps": steps, "episodes": len(episodes),
               "seconds": time.time() - t0}, open(done, "w"))
    log(f"    trained {steps} steps in {(time.time() - t0) / 60:.1f} min "
        f"({(time.time() - t0) / max(steps, 1):.2f} s/step incl. startup)")
    return loss


def evaluate(cfg, policy_dir, out_dir, seed, extra):
    """Success rate on LIBERO-Object in the simulator. Returns {'pc_success', 'raw'}."""
    res = os.path.join(out_dir + ".json")
    if os.path.exists(res):
        return json.load(open(res))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    cmd = ["lerobot-eval", f"--policy.path={policy_dir}", "--env.type=libero", "--env.task=libero_object",
           f"--eval.n_episodes={cfg.eval_episodes}", f"--eval.batch_size={cfg.eval_batch}",
           f"--seed={seed}", f"--output_dir={out_dir}", *extra]
    if cfg.eval_task_ids:
        cmd.append(f"--env.task_ids={cfg.eval_task_ids}")
    t0 = time.time()
    rc, out = run(cmd, out_dir + ".log")
    if rc != 0:
        raise RuntimeError(f"lerobot-eval failed (exit {rc}), log: {out_dir}.log\n{tail(out)}")
    info, pc = None, None
    for f in glob.glob(os.path.join(out_dir, "**", "eval_info.json"), recursive=True):
        info = json.load(open(f))
        for key in ("overall", "aggregated"):
            if isinstance(info.get(key), dict) and "pc_success" in info[key]:
                pc = info[key]["pc_success"]
    if pc is None:
        m = re.findall(r"pc_success[\"']?\s*[:=]\s*([0-9.]+)", out)
        pc = float(m[-1]) if m else None
    r = {"pc_success": pc, "seconds": time.time() - t0, "raw": info}
    json.dump(r, open(res, "w"))
    log(f"    eval: {pc}% success ({(time.time() - t0) / 60:.1f} min)")
    return r


def drop(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------- arms

def run_arm(cfg, arm, seed, sets, variant, out):
    res_path = os.path.join(out, f"seed{seed}_arm{arm}.json")
    if os.path.exists(res_path):
        log(f"seed {seed} arm {arm}: done already")
        return
    work = os.path.join(out, "work", f"seed{seed}", arm)
    stages = ARMS[arm]
    result = {"arm": arm, "seed": seed, "init": variant["name"], "config": asdict(cfg),
              "stages": [], "evals": {}}
    init = variant["base"]
    prev = None
    for k, stage in enumerate(stages):
        last = k == len(stages) - 1
        stage_dir = os.path.join(work, f"stage{k + 1}_{stage}")
        log(f"seed {seed} arm {arm}: stage {k + 1}/{len(stages)} on {stage} ({len(sets[stage])} demos)")
        args = init if prev is None else [f"--policy.path={ckpt(prev)}"]
        if last:
            if prev is not None and 0 in cfg.eval_steps:          # right after forgetting
                result["evals"]["stage3_step0"] = evaluate(cfg, ckpt(prev), os.path.join(work, "eval_s3_0"),
                                                           seed, variant["extra"])
            loss = train(cfg, args, variant["extra"], sets[stage], cfg.relearn_steps, stage_dir, seed,
                         cfg.relearn_save_freq, cfg.relearn_log_freq)
            for s in cfg.eval_steps:
                if s == 0:
                    continue
                which = "last" if s == cfg.relearn_steps else s
                result["evals"][f"stage3_step{s}"] = evaluate(cfg, ckpt(stage_dir, which),
                                                              os.path.join(work, f"eval_s3_{s}"), seed,
                                                              variant["extra"])
        else:
            loss = train(cfg, args, variant["extra"], sets[stage], cfg.stage_steps, stage_dir, seed,
                         cfg.stage_steps, cfg.log_freq)
            if arm == "A" and k == 0:                             # A's peak on Object
                result["evals"]["after_stage1"] = evaluate(cfg, ckpt(stage_dir), os.path.join(work, "eval_s1"),
                                                           seed, variant["extra"])
        result["stages"].append({"stage": k + 1, "data": stage, "demos": len(sets[stage]),
                                 "steps": cfg.relearn_steps if last else cfg.stage_steps,
                                 "log_freq": cfg.relearn_log_freq if last else cfg.log_freq, "loss": loss})
        if prev is not None:
            drop(os.path.join(prev, "checkpoints"))               # no longer needed
        prev = stage_dir
    drop(os.path.join(prev, "checkpoints"))
    json.dump(result, open(res_path, "w"), indent=1)
    log(f"seed {seed} arm {arm}: wrote {res_path}")


def choose_init(cfg, sets, out):
    """Try the ways of starting from the pretrained model; keep the first that trains AND evaluates."""
    path = os.path.join(out, "init.json")
    if os.path.exists(path):
        return json.load(open(path))
    tiny = replace(Cfg(), **SMOKE)
    eps = sets["object_pre"][:2]
    for v in INIT_VARIANTS:
        d = os.path.join(out, "init_check", v["name"])
        log(f"init check: {v['name']}")
        try:
            train(tiny, v["base"], v["extra"], eps, 10, os.path.join(d, "train"), 0, 10, 5)
            evaluate(tiny, ckpt(os.path.join(d, "train")), os.path.join(d, "eval"), 0, v["extra"])
        except RuntimeError as e:
            log(f"   {v['name']} does not work here:\n{e}")
            continue
        drop(os.path.join(d, "train", "checkpoints"))
        json.dump(v, open(path, "w"))
        log(f"init: using {v['name']}")
        return v
    raise SystemExit("no way of starting from the pretrained SmolVLA worked; see the logs in "
                     f"{os.path.join(out, 'init_check')}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--out", default=None, help="default results/vla (results/vla_smoke with --smoke)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = replace(Cfg(), **SMOKE) if args.smoke else Cfg()
    out = args.out or ("results/vla_smoke" if args.smoke else ("results/vla_full" if FULL else "results/vla"))
    if os.environ.get("VLA_BATCH"):
        cfg = replace(cfg, batch_size=int(os.environ["VLA_BATCH"]))
    log(f"training: {'WHOLE model (VLA_FULL=1)' if FULL else 'action expert only (smolvla_base default)'}, batch {cfg.batch_size}")
    os.makedirs(out, exist_ok=True)
    for tool in ("lerobot-train", "lerobot-eval"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found: activate the env_vla environment first")
    log(f"dataset {DATASET}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}, video backend {video_backend()}")
    ensure_full_dataset()
    sets = episode_sets(out)
    if args.smoke:
        sets = {k: v[:4] for k, v in sets.items()}
    variant = choose_init(cfg, sets, out)
    for seed in args.seeds:                       # every arm of a seed before the next seed
        for arm in args.arms:
            run_arm(cfg, arm, seed, sets, variant, out)
    log(f"all done -> {out}   (summarize with: python -m smolvla.report {out})")


if __name__ == "__main__":
    main()
