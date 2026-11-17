#!/usr/bin/env bash
# One-time setup for the VLA version of the experiment (SmolVLA on LIBERO), in its own conda
# environment so the TinyGPT environment is untouched. Then runs probe.py, which checks every
# piece and prints what the run script needs to know. Send the probe output back.
#
#   bash smolvla/setup.sh            (from the repo folder; ~10-20 min, mostly downloads)
#
# Everything stays under ~/ruiz: env in ~/ruiz/env_vla, caches in ~/ruiz/hf_cache.
set -uo pipefail
cd "$(dirname "$0")/.."
ENV=${VLA_ENV:-$HOME/ruiz/env_vla}
export HF_HOME=${HF_HOME:-$HOME/ruiz/hf_cache} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/ruiz/pip_cache}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p logs
LOG=logs/vla_setup_$(date +%Y%m%d-%H%M%S).log
exec > >(tee "$LOG") 2>&1

source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -d "$ENV" ]; then
  echo "== creating conda env $ENV (python 3.10)"
  conda create -y -p "$ENV" python=3.10 || { echo "conda create failed"; exit 1; }
fi
conda activate "$ENV"
echo "== python: $(which python)"
echo "== installing lerobot with the smolvla and libero extras"
pip install -q --upgrade pip
pip install -q "lerobot[smolvla,libero]" || {
  echo "== extras install failed, trying lerobot from GitHub"
  pip install -q "lerobot[smolvla,libero] @ git+https://github.com/huggingface/lerobot.git" || exit 1; }
conda install -y -q -c conda-forge ffmpeg >/dev/null 2>&1 || echo "   (ffmpeg via conda failed; video decoding may still work)"

echo "== probe"
python smolvla/probe.py
echo "== setup log: $PWD/$LOG"
