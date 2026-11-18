#!/usr/bin/env bash
# The VLA experiment in one command:   bash run_vla.sh
#
#   1. sets up its own conda env (~/ruiz/env_vla) the first time (smolvla/setup.sh)
#   2. smoke test the first time (~10-20 min): every step end to end, on a few demos; also picks
#      how to start from the pretrained SmolVLA (saved in results/vla/init.json)
#   3. the real run: arms A, C, B, S for each seed in VLA_SEEDS, one seed after another
#   4. report (results/vla/summary.md, success.png, loss.png), pack results_<id>.tgz, optional scp
#
# Detaches by itself; prints a tail -f line. Rerunning resumes (finished stages/evals/arms skipped).
# Results so far, any time:   conda activate ~/ruiz/env_vla && python -m smolvla.report results/vla
#
# Optional environment variables:
#   GPU=0                       which GPU
#   VLA_SEEDS="0 1 2"           seeds, run in this order (each seed ~6-8 h; seed 0 first)
#   RESULTS_DEST=me@host:~/runs/  scp the archive there at the end
#   VLA_FULL=1                  train the whole model, not only the action expert (results in results/vla_full)
#   VLA_BATCH=16                batch size (default 32; lower it if the whole model runs out of memory)
#   SKIP_SMOKE=1                skip the smoke test     FOREGROUND=1   do not detach
set -uo pipefail
cd "$(dirname "$0")"
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-vla}
LOGDIR="logs/$RUN_ID"
mkdir -p "$LOGDIR"
if [ -z "${FOREGROUND:-}" ] && [ -z "${_RUN_VLA_CHILD:-}" ]; then
  _RUN_VLA_CHILD=1 RUN_ID="$RUN_ID" nohup bash "$0" > "$LOGDIR/run_vla.log" 2>&1 &
  echo "Started in the background (pid $!)."
  echo "Follow it with:   tail -f $PWD/$LOGDIR/run_vla.log"
  exit 0
fi

export GPU=${GPU:-0}
export HF_HOME=${HF_HOME:-$HOME/ruiz/hf_cache} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/ruiz/pip_cache}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/ruiz/libero_config}
ENV=${VLA_ENV:-$HOME/ruiz/env_vla}
SEEDS=${VLA_SEEDS:-"0 1 2"}
OUT=results/vla; [ -n "${VLA_FULL:-}" ] && [ "${VLA_FULL}" != "0" ] && OUT=results/vla_full
echo "== $(date) run $RUN_ID on $(hostname), GPU $GPU, seeds [$SEEDS], results in $OUT"
git rev-parse --short HEAD 2>/dev/null | sed 's/^/code version: /'

echo "== setup"
bash smolvla/setup.sh 2>&1 | tee "$LOGDIR/setup.log" || true
grep -q "VLA environment ready" "$LOGDIR/setup.log" || { echo "setup failed, see $PWD/$LOGDIR/setup.log"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

mkdir -p $OUT
if [ -z "${SKIP_SMOKE:-}" ] && [ ! -f results/vla_smoke/PASSED ]; then
  echo "== smoke test (log: $PWD/$LOGDIR/vla_smoke.log)"
  if python -m smolvla.run --smoke --seeds 0 > "$LOGDIR/vla_smoke.log" 2>&1; then
    touch results/vla_smoke/PASSED
    echo "   passed; start: $(python -c "import json; print(json.load(open('results/vla_smoke/init.json'))['name'])")"
  else
    echo "   SMOKE TEST FAILED. Last lines:"; tail -40 "$LOGDIR/vla_smoke.log"; exit 1
  fi
fi
for f in init.json episodes.json; do
  [ -f $OUT/$f ] || cp results/vla_smoke/$f $OUT/$f 2>/dev/null || true
done

echo "== real run (log: $PWD/$LOGDIR/vla.log)"
python -m smolvla.run --seeds $SEEDS --out $OUT > "$LOGDIR/vla.log" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "   done" || { echo "   FAILED (exit $rc). Last lines:"; tail -30 "$LOGDIR/vla.log"; }

echo "== report"
python -m smolvla.report $OUT > "$LOGDIR/report.log" 2>&1 && cat $OUT/summary.md || echo "   report failed (see report.log)"
ARCHIVE="results_$RUN_ID.tgz"
tar czf "$ARCHIVE" --exclude='checkpoints' $OUT "$LOGDIR" 2>/dev/null
echo "== packed $PWD/$ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
if [ -n "${RESULTS_DEST:-}" ]; then
  scp -o BatchMode=yes "$ARCHIVE" "$RESULTS_DEST" && echo "   copied to $RESULTS_DEST" || echo "   copy failed; pull it instead"
fi
echo "   to pull it:   scp $(whoami)@$(hostname -f 2>/dev/null || hostname):$PWD/$ARCHIVE ."
echo "== $(date) finished"
exit $rc
