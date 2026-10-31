"""Graphs and the summary table for one results folder written by vla.py.

usage: python plots.py results/main [--levels 3.3 3.0 2.8]
writes into that folder:
  overview_seed<N>.png  every stage of arms A, B, C for one seed
  speed_seed<N>.png     stage-3 curves of every arm, lined up where each first reaches --levels[0]
  seeds.png             stage-3 curves averaged over seeds + steps between the first and last level
  summary.md            mean ± std per arm over seeds

--levels: story-test losses to time. The first one is where the speed curves are lined up,
"steps a->b" is the time from the first to the last. The defaults fit the real data;
pick others for --smoke runs.
"""
import argparse, glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, BAND = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f3f2ee"
ARMS = {"A": ("#2a78d6", "A · stories first"),
        "B": ("#eb6834", "B · code only (no stage 1)"),
        "C": ("#1baf7a", "C · Wikipedia first"),
        "S": ("#4a3aa7", "S · from scratch (stage 3 only)")}
STAGE1 = {"A": "trains on stories", "B": None, "C": "trains on Wikipedia"}

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def first_step(curve, level):
    """Step at which the curve first drops to `level` (linear between checkpoints), or None."""
    for a, b in zip(curve, curve[1:]):
        if a["pre_val"] >= level >= b["pre_val"]:
            return a["step"] + (a["pre_val"] - level) / (a["pre_val"] - b["pre_val"]) * (b["step"] - a["step"])
    return None


def load(folder):
    runs = {}
    for path in sorted(glob.glob(os.path.join(folder, "seed*_arm*.json"))):
        with open(path) as f:
            d = json.load(f)
        runs.setdefault(d["seed"], {})[d["arm"]] = d
    if not runs:
        raise SystemExit(f"no seed*_arm*.json in {folder}")
    return runs


# ---------------------------------------------------------------- metrics

def after_stage1(run):
    s1 = [r for r in run["stages12"] if r["stage"] == "1_pretrain"]
    return s1[-1]["pre_val"] if s1 else None


def metrics(runs, levels):
    """{arm: {metric: [value per seed]}}; None where a seed never got there."""
    out = {}
    for seed, arms in sorted(runs.items()):
        target = None
        if "A" in arms:
            target = after_stage1(arms["A"]) * (1 + arms["A"]["config"]["target_tol"])
        for arm, run in arms.items():
            m = out.setdefault(arm, {})
            fresh, seen = run["relearn"]["fresh"], run["relearn"]["seen"]
            start = run["probes"].get("stage3_start", {})
            vals = {
                "after stage 1": after_stage1(run),
                "stage-3 start": fresh[0]["pre_val"],
                "final (seen pile)": seen[-1]["pre_val"],
                "final (fresh pile)": fresh[-1]["pre_val"],
                "steps to target": first_step(fresh, target) if target else None,
                "seen − fresh pile at stage-3 start": (start["seen_pile"] - start["fresh_pile"]
                                                       if "seen_pile" in start else None),
            }
            for lv in levels:
                vals[f"steps to {lv}"] = first_step(fresh, lv)
            a, b = first_step(fresh, levels[0]), first_step(fresh, levels[-1])
            vals[f"steps {levels[0]}→{levels[-1]}"] = b - a if a is not None and b is not None else None
            for k, v in vals.items():
                m.setdefault(k, []).append(v)
    return out


def fmt(values, digits, steps):
    """mean ± std of the non-None values; for step counts None means 'never got there'."""
    got = [v for v in values if v is not None]
    if not got:
        return "never" if steps else "—"
    s = f"{np.mean(got):.{digits}f}"
    if len(got) > 1:
        s += f" ± {np.std(got, ddof=1):.{digits}f}"
    if steps and len(got) < len(values):
        s += f" (never in {len(values) - len(got)}/{len(values)})"
    return s


def write_summary(runs, levels, folder):
    m = metrics(runs, levels)
    arms = [a for a in ARMS if a in m]
    n = len(runs)
    cols = list(next(iter(m.values())).keys())
    lines = [f"# Summary · {n} seed(s): {', '.join(str(s) for s in sorted(runs))}", "",
             "Story-test loss (held-out TinyStories, lower = better) unless the column says steps. "
             "Mean ± sample std over seeds; no ± when n = 1. Stage-3 columns use the fresh-pile copy "
             "unless they say seen. 'steps to target' = A's after-stage-1 loss + 5%.", "",
             "| arm | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for arm in arms:
        cells = [fmt(m[arm][c], 0, True) if c.startswith("steps") else fmt(m[arm][c], 3, False) for c in cols]
        lines.append(f"| {ARMS[arm][1]} | " + " | ".join(cells) + " |")

    key = f"steps {levels[0]}→{levels[-1]}"
    if "A" in m and "C" in m:
        a, c = m["A"][key], m["C"][key]
        pairs = [(x, y) for x, y in zip(a, c) if x is not None and y is not None]
        lines += ["", f"## A vs C on '{key}' (the clean comparison: same total training, only stage-1 data differs)", ""]
        if pairs:
            diff = [y - x for x, y in pairs]
            lines.append(f"- C minus A, per seed: {', '.join(f'{d:.0f}' for d in diff)} steps")
            if len(pairs) > 1:
                sd_a, sd_c = np.std([x for x, _ in pairs], ddof=1), np.std([y for _, y in pairs], ddof=1)
                lines.append(f"- mean difference {np.mean(diff):.0f} steps; seed-to-seed std: A {sd_a:.1f}, C {sd_c:.1f}; "
                             f"difference / larger std = {np.mean(diff) / max(sd_a, sd_c, 1e-9):.1f} "
                             "(rule of thumb: above ~2 is a finding, below is a hint)")
            else:
                lines.append("- one seed only: no seed noise estimate yet, so this is a hint, not a finding")
        else:
            lines.append("- not computable: one of the arms never reached both levels")
    lines += ["", "'seen − fresh pile at stage-3 start': loss on the pile A read in stage 1 minus loss on "
              "the fresh pile, measured before stage 3. Clearly negative for A (and ~0 for B, C, S) "
              "means specific stage-1 text survived stage 2."]
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- figures

def overview(seed, arms, folder):
    rows = [a for a in ("A", "B", "C") if a in arms]
    if not rows:
        return
    pre = arms[rows[0]]["config"]["pretrain_steps"]
    orth = arms[rows[0]]["config"]["orth_steps"]
    top = max(r["pre_val"] for a in rows for r in arms[a]["stages12"]) * 1.2
    s3 = [r["pre_val"] for a in rows for sub in ("seen", "fresh") for r in arms[a]["relearn"][sub]]
    s3_hi = max(arms[a]["relearn"]["fresh"][1]["pre_val"] for a in rows) + 0.25
    s3_lo = min(s3) - 0.1

    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 3.05 * len(rows) + 0.6), width_ratios=[3, 1],
                             sharey="col", facecolor=SURFACE, layout="constrained", squeeze=False)
    for i, arm in enumerate(rows):
        ax, ax3 = axes[i]
        style(ax); style(ax3)
        log = arms[arm]["stages12"]
        shift = 0 if STAGE1[arm] else pre                # B's stage 2 lines up with the others
        ax.axvspan(0, pre, color=BAND, zorder=0)
        ax.text(pre / 2, top * 0.99, f"STAGE 1 · {STAGE1[arm]}" if STAGE1[arm] else
                "STAGE 1 · skipped (starts from random weights)",
                ha="center", va="top", color=INK2, fontsize=8.5, fontweight="bold")
        ax.text(pre + orth / 2, top * 0.99, "STAGE 2 · trains on code", ha="center", va="top",
                color=INK2, fontsize=8.5, fontweight="bold")
        ax.plot([r["global_step"] + shift for r in log], [r["pre_val"] for r in log], color=ARMS["A"][0], lw=2)
        ax.set_xlim(0, pre + orth)
        ax.set_ylim(0, top)
        ax.set_title(f"Arm {arm}", loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_ylabel("story-test loss")

        again = "trains on stories again" if arm == "A" else "trains on stories (first time)"
        ax3.set_title(f"STAGE 3 · {again}", loc="left", color=INK2, fontsize=8.5, fontweight="bold")
        for sub, ls in (("seen", "-"), ("fresh", (0, (3, 2)))):
            c = arms[arm]["relearn"][sub]
            ax3.plot([r["step"] for r in c], [r["pre_val"] for r in c], color=ARMS["A"][0], lw=2, ls=ls)
        ax3.set_ylim(s3_lo, s3_hi)
        ax3.set_xlim(0, arms[arm]["config"]["relearn_steps"])
        start = arms[arm]["relearn"]["seen"][0]["pre_val"]
        ax3.text(0.06, 0.98, f"starts at {start:.2f} ↑ (off the top)", transform=ax3.transAxes,
                 va="top", color=MUTED, fontsize=8)
        ax3.set_ylabel("story-test loss (zoomed)")
    axes[-1, 0].set_xlabel("training step")
    axes[-1, 1].set_xlabel("stage-3 step")
    handles = [Line2D([], [], color=ARMS["A"][0], lw=2,
                      label="stages 1–2, and stage-3 copy trained on the old stories (the ones A read in stage 1; new to B and C)"),
               Line2D([], [], color=ARMS["A"][0], lw=2, ls=(0, (3, 2)),
                      label="stage-3 copy trained on new stories (no arm read them before)")]
    fig.legend(handles=handles, loc="outside upper left", frameon=False, fontsize=8.5, alignment="left",
               title=f"VLA TinyGPT · seed {seed} · every line = score on the story test (stories no model "
                     "trains on); the grey band labels say what the model is training on",
               title_fontproperties={"weight": "bold", "size": 10.5})
    fig.savefig(os.path.join(folder, f"overview_seed{seed}.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def speed(seed, arms, levels, folder):
    align = levels[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=SURFACE, layout="constrained")
    style(ax)
    xmax, ends = 0, []
    for arm in [a for a in ARMS if a in arms]:
        color, label = ARMS[arm]
        curve = arms[arm]["relearn"]["fresh"]
        t0 = first_step(curve, align)
        if t0 is None:
            continue
        xs = [0.0] + [r["step"] - t0 for r in curve if r["step"] > t0]
        ys = [align] + [r["pre_val"] for r in curve if r["step"] > t0]
        ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round")
        ax.plot(xs[1:], ys[1:], "o", ms=3, color=color, mec=SURFACE, mew=0.8, zorder=4)   # real checkpoints
        ends.append((xs[-1], ys[-1], f"{label}\nreached {align} at step {t0:.0f}"))
        xmax = max(xmax, xs[-1])
    ends.sort(key=lambda e: e[1])
    for i, (x, y, text) in enumerate(ends):                 # nudge labels apart vertically
        dy = 0 if i == 0 or y - ends[i - 1][1] > 0.06 else 12
        ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)
    lo = min(ax.get_ylim()[0], align - 1.0)
    if "A" in arms and after_stage1(arms["A"]):
        target = after_stage1(arms["A"]) * (1 + arms["A"]["config"]["target_tol"])
        ax.axhline(target, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(0, target, f"recovered = {target:.2f} (A's pre-forgetting level + 5%)", va="bottom",
                color=INK2, fontsize=8.5)
        lo = target - 0.15
    ax.set_xlim(0, xmax * 1.45 if xmax else 1)
    ax.set_ylim(lo, align + 0.05)
    ax.set_xlabel(f"stage-3 steps after first reaching loss {align}")
    ax.set_ylabel("story-test loss (lower = better)")
    ax.set_title(f"Stage-3 story-learning speed from the same starting point · seed {seed} · new-stories copy",
                 loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.text(0, -0.2, f"Each curve starts when that arm first reaches loss {align}, so the head start is removed: "
            "a steeper drop = faster learning. Dots = real checkpoints.\n"
            "Only A has seen stories before stage 3; for B, C and S stage 3 is their first time.",
            transform=ax.transAxes, color=MUTED, fontsize=8)
    fig.savefig(os.path.join(folder, f"speed_seed{seed}.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def seeds_figure(runs, levels, folder):
    m = metrics(runs, levels)
    arms = [a for a in ARMS if a in m]
    key = f"steps {levels[0]}→{levels[-1]}"
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), width_ratios=[1.6, 1],
                                  facecolor=SURFACE, layout="constrained")
    style(ax); style(ax2)
    n = len(runs)
    for arm in arms:
        color, label = ARMS[arm]
        curves = [runs[s][arm]["relearn"]["fresh"] for s in sorted(runs) if arm in runs[s]]
        steps = [r["step"] for r in curves[0]]
        ys = np.array([[r["pre_val"] for r in c] for c in curves])
        ax.plot(steps, ys.mean(0), color=color, lw=2)
        if len(curves) > 1:
            ax.fill_between(steps, ys.min(0), ys.max(0), color=color, alpha=0.15, lw=0)
        ax.annotate(label, (steps[-1], ys.mean(0)[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    hi = max(runs[s][a]["relearn"]["fresh"][1]["pre_val"] for s in runs for a in arms if a in runs[s])
    ax.set_ylim(ax.get_ylim()[0], hi + 0.3)
    ax.set_xlim(0, max(r["step"] for s in runs for a in runs[s] for r in runs[s][a]["relearn"]["fresh"]) * 1.45)
    ax.set_xlabel("stage-3 step")
    ax.set_ylabel("story-test loss (zoomed; step 0 is off the top)")
    ax.set_title(f"Stage 3, new-stories copy · mean of {n} seed(s)" + (" · band = min–max" if n > 1 else ""),
                 loc="left", color=INK, fontsize=11, fontweight="bold")

    for i, arm in enumerate(arms):
        vals = [v for v in m[arm][key] if v is not None]
        color = ARMS[arm][0]
        ax2.scatter(vals, [i] * len(vals), s=40, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
        if vals:
            ax2.plot([np.mean(vals)] * 2, [i - 0.25, i + 0.25], color=INK, lw=2, zorder=4)
        missing = len(m[arm][key]) - len(vals)
        if missing:
            ax2.text(0, i + 0.3, f"{missing} seed(s) never reached {levels[-1]}", fontsize=8, color=MUTED)
    ax2.set_yticks(range(len(arms)), [ARMS[a][1] for a in arms])
    ax2.invert_yaxis()
    ax2.set_xlim(0, None)
    ax2.grid(axis="x", color=GRID, lw=0.6)
    ax2.set_xlabel(f"steps to go from loss {levels[0]} to {levels[-1]} (fewer = faster)")
    ax2.set_title("Learning speed per seed (dots) and mean (bar)", loc="left", color=INK, fontsize=11,
                  fontweight="bold")
    fig.savefig(os.path.join(folder, "seeds.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--levels", type=float, nargs="+", default=[3.3, 3.0, 2.8])
    args = p.parse_args()
    sys.stdout.reconfigure(errors="replace")   # Windows consoles can't print ± or →
    runs = load(args.folder)
    for seed, arms in sorted(runs.items()):
        overview(seed, arms, args.folder)
        speed(seed, arms, args.levels, args.folder)
    seeds_figure(runs, args.levels, args.folder)
    write_summary(runs, args.levels, args.folder)
    print(f"\nwrote figures + summary.md to {args.folder}")


if __name__ == "__main__":
    main()
