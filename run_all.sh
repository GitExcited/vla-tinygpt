#!/usr/bin/env bash
# Everything in one command:   bash run_all.sh
#
# Runs, in parallel on one GPU:
#   - TinyGPT v2 ("done properly"): 4 arms x TINY_SEEDS, stage 3 from five stage-2 depths
#   - hidden/: gone-or-hidden on synthetic biographies (main model + never-had-it twin) x HIDDEN_SEEDS
# then writes the reports, packs everything into results_<RUN_ID>.tgz, and (optionally) copies it
# to RESULTS_DEST with scp.
#
# Detaches by itself (nohup), so you can log out. Follow it with the `tail -f` line it prints.
# Rerunning resumes: finished (seed, arm) / (seed, condition) files are skipped.
#
# Optional environment variables:
#   GPU=0                         which GPU to use
#   TINY_SEEDS="0 1 2 3 4"        TinyGPT seeds (each one is a separate process)
#   HIDDEN_SEEDS="0 1 2"          gone-or-hidden seeds
#   RESULTS_DEST=me@host:~/runs/  copy the results archive there at the end (needs ssh access to it)
#   SMOKE=1                       tiny version of everything (a few minutes) to check the setup first
#   FOREGROUND=1                  do not detach
#   PYTHON=python3                which python to use
#   SKIP_PIP=1                    do not pip install requirements.txt
#
# Example:   SMOKE=1 bash run_all.sh      # check first
#            RESULTS_DEST=me@mylaptop:~/runs/ bash run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)${SMOKE:+-smoke}}
LOGDIR="logs/$RUN_ID"
mkdir -p "$LOGDIR"

if [ -z "${FOREGROUND:-}" ] && [ -z "${_RUN_ALL_CHILD:-}" ]; then
  _RUN_ALL_CHILD=1 RUN_ID="$RUN_ID" nohup bash "$0" > "$LOGDIR/run_all.log" 2>&1 &
  echo "Started in the background (pid $!)."
  echo "Follow it with:   tail -f $PWD/$LOGDIR/run_all.log"
  exit 0
fi

PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1
if [ -n "${SMOKE:-}" ]; then
  TINY_SEEDS=${TINY_SEEDS:-0}; HIDDEN_SEEDS=${HIDDEN_SEEDS:-0}
  TINY_FLAGS="--smoke"; HIDDEN_FLAGS="--smoke"; TINY_TAG=v2_smoke; HIDDEN_TAG=hidden_smoke
  LEVELS="--levels 3.0 2.0 1.7"
else
  TINY_SEEDS=${TINY_SEEDS:-"0 1 2 3 4"}; HIDDEN_SEEDS=${HIDDEN_SEEDS:-"0 1 2"}
  TINY_FLAGS=""; HIDDEN_FLAGS=""; TINY_TAG=v2; HIDDEN_TAG=hidden
  LEVELS=""
fi

echo "== $(date) run $RUN_ID on $(hostname)"
echo "== setup"
if [ -z "${SKIP_PIP:-}" ]; then
  $PY -m pip install -q -r requirements.txt || { echo "pip install failed"; exit 1; }
fi
$PY -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if ok else 'NONE (CPU only)')"
git rev-parse --short HEAD 2>/dev/null | sed 's/^/code version: /'

if [ -z "${SMOKE:-}" ]; then
  echo "== speed test (TinyGPT v2 model; 2 minutes)"
  $PY bench.py --untied | tee "$LOGDIR/bench.txt"
fi

echo "== data (downloads and tokenizes on the first run only)"
$PY vla.py --preset v2 $TINY_FLAGS --prepare-only || { echo "data preparation failed"; exit 1; }

echo "== launching: TinyGPT seeds [$TINY_SEEDS], gone-or-hidden seeds [$HIDDEN_SEEDS]"
pids=(); names=()
for s in $TINY_SEEDS; do
  $PY vla.py --preset v2 $TINY_FLAGS --seeds "$s" --tag "$TINY_TAG" > "$LOGDIR/tinygpt_seed$s.log" 2>&1 &
  pids+=($!); names+=("tinygpt seed $s")
done
for s in $HIDDEN_SEEDS; do
  $PY -m hidden.run $HIDDEN_FLAGS --seeds "$s" --out "results/$HIDDEN_TAG" > "$LOGDIR/hidden_seed$s.log" 2>&1 &
  pids+=($!); names+=("hidden seed $s")
done
echo "   logs: $PWD/$LOGDIR/*.log"
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "   done: ${names[$i]}"; else echo "   FAILED: ${names[$i]} (see its log)"; failed=1; fi
done

echo "== reports"
$PY plots.py "results/$TINY_TAG" $LEVELS > "$LOGDIR/plots.log" 2>&1 || echo "   plots.py failed (see plots.log)"
$PY depths.py "results/$TINY_TAG" > "$LOGDIR/depths.log" 2>&1 || echo "   depths.py failed (see depths.log)"
$PY -m hidden.report "results/$HIDDEN_TAG" > "$LOGDIR/hidden_report.log" 2>&1 || echo "   hidden.report failed (see hidden_report.log)"

ARCHIVE="results_$RUN_ID.tgz"
tar czf "$ARCHIVE" "results/$TINY_TAG" "results/$HIDDEN_TAG" "$LOGDIR" 2>/dev/null
echo "== packed $PWD/$ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
if [ -n "${RESULTS_DEST:-}" ]; then
  if scp -o BatchMode=yes "$ARCHIVE" "$RESULTS_DEST"; then echo "   copied to $RESULTS_DEST"
  else echo "   copy to $RESULTS_DEST failed; pull it instead (below)"; fi
fi
echo "   to pull it from your own machine:   scp $(whoami)@$(hostname -f 2>/dev/null || hostname):$PWD/$ARCHIVE ."
[ $failed -eq 0 ] && echo "== all done $(date)" || echo "== finished WITH FAILURES $(date)"
exit $failed
