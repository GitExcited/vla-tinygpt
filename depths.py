"""v2 analysis: does A's relearning lead over the never-had-it arms shrink with more stage 2?

For every stage-2 depth (the step of code training that stage 3 started from) and every arm,
it reads the fresh-pile stage-3 curve (story-test loss) and reports, as mean +- std over seeds:
  - the loss at a few stage-3 steps, and
  - each arm's gap to A at those steps (positive = A is ahead).
Writes depths.md and depths.png into the results folder.

usage: python depths.py results/v2
"""
import glob, json, os, sys

import numpy as np

STEPS = [0, 1, 2, 5, 10, 25, 50, 100, 250, 500]


def load(folder):
    runs = {}
    for path in sorted(glob.glob(os.path.join(folder, "seed*_arm*.json"))):
        with open(path) as f:
            d = json.load(f)
        if "relearn_by_depth" in d:
            runs.setdefault(d["seed"], {})[d["arm"]] = d
    if not runs:
        raise SystemExit(f"no v2 results (relearn_by_depth) in {folder}")
    return runs


def loss_at(curve, step):
    for r in curve:
        if r["step"] == step:
            return r["pre_val"]
    return None


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    return f"{np.mean(xs):.2f} ± {np.std(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/v2"
    runs = load(folder)
    seeds = sorted(runs)
    any_run = next(iter(runs[seeds[0]].values()))
    depths = [int(d) for d in any_run["config"]["stage2_depths"].split(",")]
    final = depths[-1]
    arms = [a for a in ("A", "C", "B", "S") if any(a in runs[s] for s in seeds)]

    def curve(seed, arm, depth):
        d = runs[seed].get(arm)
        if d is None:
            return None
        key = str(final if arm == "S" else depth)        # S has no stage 2
        return d["relearn_by_depth"].get(key, {}).get("fresh")

    lines = [f"# Relearning by stage-2 depth ({len(seeds)} seeds: {seeds})", "",
             "Story-test loss on the fresh pile in stage 3, mean ± std over seeds. "
             "Gap = arm minus A (positive: A is ahead). S has no stage 2, so its row is the same at every depth.", ""]
    gaps = {}
    for depth in depths:
        lines += [f"## Stage 3 started after {depth} stage-2 steps", "",
                  "| | " + " | ".join(f"step {s}" for s in STEPS) + " |",
                  "|---|" + "---|" * len(STEPS)]
        for arm in arms:
            vals = [[loss_at(curve(s, arm, depth) or [], st) for s in seeds] for st in STEPS]
            lines.append(f"| {arm} loss | " + " | ".join(fmt(v) for v in vals) + " |")
        for arm in arms:
            if arm == "A":
                continue
            row = []
            for st in STEPS:
                g = []
                for s in seeds:
                    a, b = loss_at(curve(s, "A", depth) or [], st), loss_at(curve(s, arm, depth) or [], st)
                    g.append(b - a if a is not None and b is not None else None)
                row.append(g)
                gaps.setdefault(arm, {}).setdefault(st, {})[depth] = g
            lines.append(f"| {arm} - A | " + " | ".join(fmt(v) for v in row) + " |")
        lines.append("")
    with open(os.path.join(folder, "depths.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    show = [st for st in (0, 10, 50, 500)
            if any(x is not None for arm in gaps for d in depths for x in gaps[arm][st][d])]
    if not show:
        return
    fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 3.2), sharey=True)
    for ax, st in zip(np.atleast_1d(axes), show):
        for arm, color in (("C", "#1f77b4"), ("B", "#ff7f0e"), ("S", "#7f7f7f")):
            if arm not in gaps:
                continue
            m = [np.mean([x for x in gaps[arm][st][d] if x is not None] or [np.nan]) for d in depths]
            sd = [np.std([x for x in gaps[arm][st][d] if x is not None] or [np.nan]) for d in depths]
            ax.errorbar(depths, m, yerr=sd, marker="o", capsize=3, color=color, label=f"{arm} - A")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"stage-3 step {st}")
        ax.set_xlabel("stage-2 steps before relearning")
    np.atleast_1d(axes)[0].set_ylabel("loss gap to A (story test)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "depths.png"), dpi=150)


if __name__ == "__main__":
    main()
