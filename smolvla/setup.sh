#!/usr/bin/env bash
# One-time setup for the VLA experiment (SmolVLA on LIBERO), in its own conda environment so the
# TinyGPT environment is untouched. Safe to rerun: skips what is already installed.
# run_vla.sh calls it automatically; run it alone only to debug the install.
#
#   bash smolvla/setup.sh            (PROBE=1 also runs smolvla/probe.py, a long diagnostic)
#
# Everything stays under ~/ruiz: env in ~/ruiz/env_vla, caches in ~/ruiz/hf_cache.
set -uo pipefail
cd "$(dirname "$0")/.."
ENV=${VLA_ENV:-$HOME/ruiz/env_vla}
export HF_HOME=${HF_HOME:-$HOME/ruiz/hf_cache} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/ruiz/pip_cache}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=${GPU:-0}

source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -d "$ENV" ]; then
  echo "== creating conda env $ENV (python 3.10)"
  conda create -y -p "$ENV" python=3.10 || { echo "conda create failed"; exit 1; }
fi
conda activate "$ENV"
echo "== python: $(which python)"

if ! python -c "import lerobot, libero" 2>/dev/null; then
  echo "== installing lerobot with the smolvla and libero extras (a few minutes)"
  pip install -q --upgrade pip
  pip install -q "lerobot[smolvla,libero]" || {
    echo "== PyPI install failed, trying lerobot from GitHub"
    pip install -q "lerobot[smolvla,libero] @ git+https://github.com/huggingface/lerobot.git" || exit 1; }
  conda install -y -q -c conda-forge ffmpeg >/dev/null 2>&1 || echo "   (ffmpeg via conda failed; continuing)"
fi

echo "== GPU check"
if ! python -c "import torch; x=torch.randn(512,512,device='cuda'); (x@x).sum().item(); print('   torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"; then
  echo "== this torch build cannot run on the GPU; installing a build for Blackwell (CUDA 12.8)"
  pip install -q --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 || exit 1
  python -c "import torch; x=torch.randn(512,512,device='cuda'); (x@x).sum().item(); print('   torch', torch.__version__, 'OK')" || exit 1
fi
python -c "import lerobot; print('   lerobot', getattr(lerobot, '__version__', '?'))"
command -v lerobot-train >/dev/null && command -v lerobot-eval >/dev/null || { echo "lerobot-train/eval missing"; exit 1; }

if [ -n "${PROBE:-}" ]; then
  echo "== probe"
  python smolvla/probe.py
fi
echo "== VLA environment ready: conda activate $ENV"
