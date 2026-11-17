"""Summarize hidden.run results: summary.md and three figures.

  finetune.png   old-fact accuracy during fine-tuning (Palit's Fig. 1), main vs twin
  untaught.png   THE test: accuracy on the untaught half of A while relearning only the taught
                 half, starting from each fine-tuning checkpoint; main vs twin (never had A)
  edit.png       old-fact accuracy after removing the top-k parts of the weight update vs
                 random parts of the same size, at each checkpoint

usage: python -m hidden.report results/hidden
"""
import glob, json, os, sys

import numpy as np


def load(folder):
    runs = {}
    for path in sorted(glob.glob(os.path.join(folder, "seed*_*.json"))):
        with open(path) as f:
            d = json.load(f)
        runs.setdefault(d["cond"], []).append(d)
    if not runs:
        raise SystemExit(f"no seed*_*.json in {folder}")
    return runs


def ms(xs):
    xs = np.array([x for x in xs if x is not None], dtype=float)
    return (np.nan, np.nan) if xs.size == 0 else (xs.mean(), xs.std())


def fmt(xs):
    m, s = ms(xs)
    return "–" if np.isnan(m) else (f"{m:.2f}" if len(xs) < 2 else f"{m:.2f}±{s:.2f}")


def relearn_at(rec, k, name):
    for row in rec["relearn"]:
        if row["step"] == k:
            return row[name]["acc"]
    return None


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/hidden"
    runs = load(folder)
    cfg = next(iter(runs.values()))[0]["config"]
    ckpts = [str(t) for t in cfg["ckpt_steps"]]
    ks = [k for k in (0, 10, 50, cfg["relearn_steps"]) if k in cfg["relearn_eval"]]
    lines = ["# Gone or hidden? Results", "",
             f"Seeds per condition: { {c: len(v) for c, v in runs.items()} }. Values are mean±std over seeds.",
             "Accuracy = full-vocabulary argmax on the answer token, held-out phrasings.", ""]

    for cond, rs in runs.items():
        lines += [f"## Condition: {cond}", "",
                  "### Untaught half of A while relearning the taught half",
                  "main = had A, then fine-tuned; twin = never had A (same everything else). "
                  "The untaught half is never trained on during relearning.", "",
                  "| fine-tune step | old acc (main) | " +
                  " | ".join(f"untaught @{k}: main / twin" for k in ks) + " | " +
                  " | ".join(f"taught @{k}: main / twin" for k in ks if k) + " |",
                  "|---|---|" + "---|" * (len(ks) + len([k for k in ks if k]))]
        for t in ckpts:
            old = fmt([r["main"]["checkpoints"][t]["edit"]["none"]["old"]["acc"] for r in rs])
            cells = [f"{fmt([relearn_at(r['main']['checkpoints'][t], k, 'A_untaught') for r in rs])} / "
                     f"{fmt([relearn_at(r['twin']['checkpoints'][t], k, 'A_untaught') for r in rs])}" for k in ks]
            cells += [f"{fmt([relearn_at(r['main']['checkpoints'][t], k, 'A_taught') for r in rs])} / "
                      f"{fmt([relearn_at(r['twin']['checkpoints'][t], k, 'A_taught') for r in rs])}" for k in ks if k]
            lines.append(f"| {t} | {old} | " + " | ".join(cells) + " |")
        edits = [e for e in rs[0]["main"]["checkpoints"][ckpts[0]]["edit"] if e != "none"]
        lines += ["", "### Edits (main model, no training): old-fact accuracy / new-fact accuracy", "",
                  "| fine-tune step | none | " + " | ".join(edits) + " | common part removed |",
                  "|---|---|" + "---|" * (len(edits) + 1)]
        for t in ckpts:
            def cell(e):
                return (fmt([r["main"]["checkpoints"][t]["edit"][e]["old"]["acc"] for r in rs]) + " / " +
                        fmt([r["main"]["checkpoints"][t]["edit"][e]["new"]["acc"] for r in rs]))
            cp = fmt([r["main"]["checkpoints"][t]["common_part"]["common_removed"]["acc"] for r in rs])
            lines.append(f"| {t} | {cell('none')} | " + " | ".join(cell(e) for e in edits) + f" | {cp} |")
        lines.append("")
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(folder, 'summary.md')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    conds = list(runs)
    steps_ck = [int(t) for t in ckpts]

    # 1. fine-tuning curves
    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 3.4), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        rs = runs[cond]
        x = [row["step"] for row in rs[0]["main"]["finetune"]]
        for variant, key, label, col in (("main", "A", "main: A (old facts)", "C3"),
                                         ("twin", "A_twin", "twin: A_twin (its old facts)", "C7"),
                                         ("main", "ballast", "main: ballast", "C0"),
                                         ("main", "new", "main: new facts", "C2")):
            y = np.array([[row[key]["acc"] for row in r[variant]["finetune"]] for r in rs])
            ax.plot(x, y.mean(0), color=col, label=label)
            ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=col, alpha=0.2)
        ax.set_title(f"fine-tuning on new facts ({cond})")
        ax.set_xlabel("fine-tune step"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(folder, "finetune.png"), dpi=150); plt.close(fig)

    # 2. untaught half
    kplot = [k for k in ks if k][:3]
    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 3.4), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        rs = runs[cond]
        old = np.array([[r["main"]["checkpoints"][t]["edit"]["none"]["old"]["acc"] for t in ckpts] for r in rs])
        ax.plot(steps_ck, old.mean(0), "k--", label="main, before relearning")
        for i, k in enumerate(kplot):
            for variant, ls in (("main", "-"), ("twin", ":")):
                y = np.array([[relearn_at(r[variant]["checkpoints"][t], k, "A_untaught") for t in ckpts] for r in rs],
                             dtype=float)
                ax.plot(steps_ck, np.nanmean(y, 0), ls, color=f"C{i}", marker="o", ms=3,
                        label=f"{variant}, after {k} relearning steps")
        ax.set_title(f"untaught half of A ({cond})")
        ax.set_xlabel("fine-tune step the relearning started from"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(folder, "untaught.png"), dpi=150); plt.close(fig)

    # 3. edits
    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 3.4), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        rs = runs[cond]
        ranks = cfg["edit_ranks"]
        for e, col in [("none", "k")] + [(f"top{k}", f"C{i}") for i, k in enumerate(ranks)] + \
                      [(f"random{ranks[-1]}", "C7")]:
            y = np.array([[r["main"]["checkpoints"][t]["edit"][e]["old"]["acc"] for t in ckpts] for r in rs])
            ax.plot(steps_ck, y.mean(0), color=col, marker="o", ms=3, ls="--" if e.startswith("random") else "-",
                    label=e)
        ax.set_title(f"old facts after editing the update ({cond})")
        ax.set_xlabel("fine-tune step"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(folder, "edit.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
