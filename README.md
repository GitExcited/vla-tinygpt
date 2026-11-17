# vla-tinygpt

Does a small language model that has *forgotten* a domain relearn it faster than one that never learned it? A 30M-parameter GPT, three training stages, four arms.

| Arm | Stage 1 (5,000 steps) | Stage 2 (5,000 steps) | Stage 3 (500 steps) |
|---|---|---|---|
| **A** | TinyStories | code | TinyStories again |
| **B** | — | code | TinyStories (first time) |
| **C** | WikiText-103 | code | TinyStories (first time) |
| **S** | — | — | TinyStories (from scratch) |

- **Shared setup.** All arms start from the same random weights for a given seed, and use the same code data and stage-3 data.
- **Comparisons.**
  - **A vs C** is the clean one: same amount of training, and only the stage-1 data differs.
  - **B** has had less training than A and C.
  - **S** is the from-scratch baseline.
- **Two stage-3 copies.** Stage 3 runs twice from the same weights:
  - **seen pile:** the first 1M tokens of A's stage-1 data.
  - **fresh pile:** stories no arm has trained on.
- **Scoring.** Every curve is scored on held-out TinyStories (the "story test").
- **Memorization probe.** At the start and end of stage 3, the seen and fresh piles are also scored directly. If A is better on the seen pile than on the fresh one, specific stage-1 text survived stage 2, not just the language.

## Everything in one command (Nov 2026)

```bash
git clone https://github.com/GitExcited/vla-tinygpt.git && cd vla-tinygpt
SMOKE=1 bash run_all.sh                                # a few minutes: checks the whole pipeline
RESULTS_DEST=me@myhost:~/runs/ bash run_all.sh         # the real thing; detaches, prints a tail -f line
```

`run_all.sh` runs two experiments side by side on one GPU, writes the reports, packs
`results_<run id>.tgz`, and copies it to `RESULTS_DEST` with `scp` if you set it (otherwise it
prints the `scp` line to pull it). Rerunning resumes where it stopped. Options are listed at the
top of the script (`GPU`, `TINY_SEEDS`, `HIDDEN_SEEDS`, ...).

1. **TinyGPT v2** (`vla.py --preset v2`, results in `results/v2/`). The experiment above, done
   properly: 5 seeds, untied token table, story loss scored at every step for the first 50
   stage-3 steps, and stage 3 run from five points of stage 2 (after 100, 500, 1,000, 2,000 and
   5,000 code steps) to see whether A's lead shrinks the longer it trains on code. One pass over
   a ~4.1M-token relearning pile. `depths.py results/v2` writes `depths.md` / `depths.png`.
   Disk: the stage-2 snapshots take about 1 GB per (seed, arm), ~15 GB in total, in `checkpoints/`.
2. **Gone or hidden?** (`hidden/`, results in `results/hidden/`). Synthetic biographies in the
   style of Palit et al. 2026 (arXiv 2610.08718), reimplemented here with our own names,
   answers and templates: an 8-layer attention-only transformer learns 2,000 people (A) plus
   2,000 others, then is fine-tuned on 500 new people with no replay. A **twin** is trained
   identically except that A is replaced by 2,000 different people. From checkpoints along the
   fine-tuning, both relearn only the first half of A; the second half is never shown, so its
   score can only come back if its facts were still stored. Also: removing the top parts of the
   weight update (vs random parts), and Palit's logit common-part removal.
   `python -m hidden.report results/hidden` writes `summary.md`, `finetune.png`,
   `untaught.png` and `edit.png`.

## Run (v1)

```bash
pip install -r requirements.txt
python bench.py                                  # 2-minute speed test on this GPU
python vla.py --seeds 0 1 2 --mem-frac 0.33      # full run: data download on first run, then ~4 arms x 3 seeds
python plots.py results/main                     # figures + summary.md into results/main/
```

- **Choosing a GPU:** prefix commands with `CUDA_VISIBLE_DEVICES=1` (or `0`).
- **Restarts:** a finished (seed, arm) is skipped, so just rerun the same command.
- **Quick test:** `--smoke` runs a tiny synthetic version that works on CPU. Plot it with `--levels 3.0 2.0 1.7`.

### Stage-3 variants without retraining

Each arm's end-of-stage-2 model is saved in `checkpoints/`. Any run whose stage-1 and stage-2 settings match reuses it and only redoes stage 3 (minutes, not hours):

```bash
python vla.py --seeds 0 1 2 --tag onepass  --set relearn_steps=120       # one pass over the 1M-token pile
python vla.py --seeds 0 1 2 --tag slow3    --set relearn_lr=1e-4
python vla.py --seeds 0 1 2 --tag gentle2  --set orth_lr=1e-4            # changes stage 2 -> retrains, new checkpoints
```

`--set` takes any field of `Config` in `vla.py`. Changing `relearn_tokens` re-downloads the data, but stage-1 data stays the same, so checkpoints are still reused.

## Outputs (`results/<tag>/`)

- `seed<N>_arm<X>.json`: every evaluation for one seed and arm:
  - `stages12`: the stage-1 and stage-2 curves.
  - `relearn.seen` / `relearn.fresh`: the two stage-3 curves.
  - `probes`: losses on the story test, seen pile and fresh pile after stage 1, at the start of stage 3, and at the end of each stage-3 copy.
  - `config`: the settings used.
- `overview_seed<N>.png`: every stage of A, B and C.
- `speed_seed<N>.png`: stage-3 curves lined up where each arm first reaches the same loss, so the head start is removed.
- `seeds.png`: stage-3 curves averaged over seeds, and steps between two loss levels per seed.
- `summary.md`: mean ± std per arm. It also includes an A-vs-C difference compared with the seed-to-seed spread.

## Known limits

- **The relearning pile repeats.** 500 stage-3 steps loop over the 1M-token pile about 4 times. Late stage-3 numbers partly measure overfitting to that pile.
- **Stage 2 is not a gentle fine-tune.** It uses the same learning rate and length as stage 1, which is why A forgets so much. `--set orth_lr=...` tests a gentler version.
