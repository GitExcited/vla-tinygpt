"""Check every piece the SmolVLA-on-LIBERO run needs, and print what the run script must know.
Each check runs on its own; a failure prints a traceback and the next check still runs.

  python smolvla/probe.py          (inside the env_vla environment)
"""
import importlib, os, subprocess, sys, time, traceback

DATASET = os.environ.get("VLA_DATASET", "lerobot/libero")
POLICY = os.environ.get("VLA_POLICY", "lerobot/smolvla_base")
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]


def check(name, fn):
    print(f"\n== {name}", flush=True)
    t0 = time.time()
    try:
        out = fn()
        print(f"   OK ({time.time() - t0:.0f}s)" + (f": {out}" if out is not None else ""), flush=True)
        return out
    except Exception:
        traceback.print_exc()
        print(f"   FAILED ({time.time() - t0:.0f}s)", flush=True)


def first_import(*names):
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError:
            continue
    raise ImportError(f"none of {names}")


def versions():
    import torch, lerobot
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"
    x = torch.randn(1000, 1000, device="cuda") if torch.cuda.is_available() else None
    return (f"lerobot {getattr(lerobot, '__version__', '?')}, torch {torch.__version__}, {gpu}"
            + (f", matmul ok {float((x @ x).sum()):.1f}" if x is not None else ""))


def dataset_meta():
    m = first_import("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset")
    meta = m.LeRobotDatasetMetadata(DATASET)
    print(f"   episodes {meta.total_episodes}, frames {meta.total_frames}, fps {meta.fps}")
    print(f"   features: {list(meta.features)}")
    tasks = meta.tasks
    print(f"   tasks object: {type(tasks).__name__}, {len(tasks)} tasks")
    print("   first tasks:", str(tasks)[:1500])
    eps = getattr(meta, "episodes", None)
    print(f"   episodes object: {type(eps).__name__}")
    print("   sample:", str(eps)[:1500] if eps is not None else None)
    return "metadata loaded"


def libero_suites():
    bench = first_import("libero.libero.benchmark")
    d = bench.get_benchmark_dict()
    for s in SUITES:
        suite = d[s]()
        names = [suite.get_task(i).language for i in range(suite.n_tasks)]
        print(f"   {s}: {suite.n_tasks} tasks")
        for n in names:
            print(f"      - {n}")
    return "suites listed"


def policy_load():
    import torch
    m = first_import("lerobot.policies.smolvla.modeling_smolvla", "lerobot.common.policies.smolvla.modeling_smolvla")
    p = m.SmolVLAPolicy.from_pretrained(POLICY)
    n = sum(x.numel() for x in p.parameters())
    nt = sum(x.numel() for x in p.parameters() if x.requires_grad)
    if torch.cuda.is_available():
        p = p.cuda()
        print(f"   GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    cfg = p.config
    for k in ("chunk_size", "n_action_steps", "n_obs_steps", "freeze_vision_encoder",
              "train_expert_only", "optimizer_lr", "input_features", "output_features"):
        if hasattr(cfg, k):
            print(f"   config.{k} = {getattr(cfg, k)}")
    return f"{n / 1e6:.0f}M params, {nt / 1e6:.0f}M trainable"


def cli(args, timeout):
    print("   $ " + " ".join(args), flush=True)
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout + r.stderr).strip().splitlines()[-25:]
    print("\n".join("   | " + l for l in tail))
    if r.returncode != 0:
        raise RuntimeError(f"exit code {r.returncode}")


def train_smoke():
    out = os.path.join("logs", "vla_probe_train")
    subprocess.run(["rm", "-rf", out])
    cli(["lerobot-train", f"--policy.path={POLICY}", f"--dataset.repo_id={DATASET}",
         "--dataset.episodes=[0,1,2,3]", "--batch_size=8", "--steps=30", "--log_freq=10",
         "--save_freq=30", "--eval_freq=0", f"--output_dir={out}", "--policy.push_to_hub=false",
         "--wandb.enable=false"], timeout=3600)
    return f"trained 30 steps, checkpoint in {out}"


def eval_smoke():
    out = os.path.join("logs", "vla_probe_eval")
    ck = os.path.join("logs", "vla_probe_train", "checkpoints", "last", "pretrained_model")
    cli(["lerobot-eval", f"--policy.path={ck}", "--env.type=libero", "--env.task=libero_object",
         "--eval.batch_size=1", "--eval.n_episodes=2", f"--output_dir={out}"], timeout=3600)
    return "2 simulated episodes ran"


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}, HF_HOME={os.environ.get('HF_HOME')}")
    check("versions + GPU", versions)
    check(f"dataset metadata ({DATASET})", dataset_meta)
    check("LIBERO suites and task names", libero_suites)
    check(f"policy load ({POLICY})", policy_load)
    check("training smoke test (lerobot-train, 30 steps)", train_smoke)
    check("simulator smoke test (lerobot-eval, 2 episodes)", eval_smoke)
    print("\n== probe finished. Send this whole output back.")
