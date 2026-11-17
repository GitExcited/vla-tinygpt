"""Summarize smolvla.run results: summary.md, success.png and loss.png.

  success.png  success rate on LIBERO-Object during stage 3 (relearning Object), per arm
  loss.png     stage-3 training loss on demos no arm had seen (first pass), per arm

usage: python -m smolvla.report results/vla
"""
import glob, json, os, sys

import numpy as np

ARMS = ["A", "C", "B", "S"]
LABEL = {"A": "A: had Object, then Goal", "C": "C: had Spatial, then Goal (twin)",
         "B": "B: only Goal", "S": "S: pretrained SmolVLA only"}
COLOR = {"A": "C3", "C": "C0", "B": "C1", "S": "C7"}


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/vla"
    runs = {}
    for f in sorted(glob.glob(os.path.join(folder, "seed*_arm*.json"))):
        d = json.load(open(f))
        runs.setdefault(d["arm"], []).append(d)
    if not runs:
        raise SystemExit(f"no seed*_arm*.json in {folder}")
    any_run = next(iter(runs.values()))[0]
    steps = any_run["config"]["eval_steps"]
    arms = [a for a in ARMS if a in runs]

    def sr(d, s):
        e = d["evals"].get(f"stage3_step{s}")
        return None if e is None else e["pc_success"]

    def fmt(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return "–"
        return f"{np.mean(xs):.0f}" + (f" ± {np.std(xs):.0f}" if len(xs) > 1 else "")

    lines = ["# VLA version (SmolVLA on LIBERO): results", "",
             f"Start: `{any_run['init']}`. Seeds per arm: { {a: len(runs[a]) for a in arms} }.", "",
             "## Success rate on LIBERO-Object (%), during stage 3", "",
             "| arm | after stage 1 | " + " | ".join(f"stage-3 step {s}" for s in steps) + " |",
             "|---|---|" + "---|" * len(steps)]
    for a in arms:
        s1 = fmt([d["evals"].get("after_stage1", {}).get("pc_success") for d in runs[a]])
        lines.append(f"| {LABEL[a]} | {s1} | " + " | ".join(fmt([sr(d, s) for d in runs[a]]) for s in steps) + " |")
    lines += ["", "A vs C is the comparison that matters: same amount of training, only the stage-1 suite differs.",
              "Step 0 = right after stage 2 (forgetting); S has no step 0.", "",
              "## Stage-3 training loss (first pass over unseen Object demos)", "",
              "| arm | " + " | ".join(f"step {k}" for k in (5, 25, 50, 100, 250, 500)) + " |",
              "|---|" + "---|" * 6]
    for a in arms:
        cells = []
        for k in (5, 25, 50, 100, 250, 500):
            vals = []
            for d in runs[a]:
                st = d["stages"][-1]
                i = k // st["log_freq"] - 1
                if 0 <= i < len(st["loss"]):
                    vals.append(st["loss"][i])
            cells.append("–" if not vals else f"{np.mean(vals):.3f}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for a in arms:
        ys = np.array([[np.nan if sr(d, s) is None else sr(d, s) for s in steps] for d in runs[a]], dtype=float)
        ax.plot(steps, np.nanmean(ys, 0), marker="o", color=COLOR[a], label=LABEL[a])
    ax.set_xlabel("stage-3 step (relearning LIBERO-Object)")
    ax.set_ylabel("success rate on LIBERO-Object (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(folder, "success.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for a in arms:
        curves = []
        for d in runs[a]:
            st = d["stages"][-1]
            curves.append(np.array(st["loss"], dtype=float))
        n = min(len(c) for c in curves)
        if n == 0:
            continue
        y = np.mean([c[:n] for c in curves], 0)
        x = (np.arange(n) + 1) * runs[a][0]["stages"][-1]["log_freq"]
        ax.plot(x, y, color=COLOR[a], label=LABEL[a], lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("stage-3 step")
    ax.set_ylabel("training loss (unseen demos, first pass)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(folder, "loss.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
