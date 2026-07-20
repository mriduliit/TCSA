#!/usr/bin/env bash
# setup_env.sh — bootstrap the sd3_gen conda environment
#
# Usage:
#   bash sd3/setup_env.sh            # create & install
#   bash sd3/setup_env.sh --rebuild  # nuke and recreate
#
# After setup:
#   conda activate sd3_gen
#   python sd3/gen_images.py --prompts "a sunset over the ocean" --out_dir sd3/outputs

set -euo pipefail

ENV="sd3_gen"
PY="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── parse flags ──────────────────────────────────────────────────────────────
REBUILD=0
for arg in "$@"; do
  [[ "$arg" == "--rebuild" ]] && REBUILD=1
done

# ── optional teardown ────────────────────────────────────────────────────────
if [[ $REBUILD -eq 1 ]]; then
  echo "[setup] Removing existing env: $ENV"
  conda env remove -n "$ENV" -y 2>/dev/null || true
fi

# ── create env if missing ────────────────────────────────────────────────────
if conda env list | grep -qE "^${ENV}\s"; then
  echo "[setup] Env '$ENV' already exists — skipping creation (use --rebuild to reset)"
else
  echo "[setup] Creating conda env '$ENV'  python=$PY"
  conda create -y -n "$ENV" python="$PY"
fi

# ── PyTorch (CUDA 12.4) ──────────────────────────────────────────────────────
echo "[setup] Installing PyTorch 2.6 + CUDA 12.4"
conda run --no-capture-output -n "$ENV" \
  pip install --quiet \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# ── SD3 / diffusers stack ────────────────────────────────────────────────────
echo "[setup] Installing SD3 requirements from $SCRIPT_DIR/requirements.txt"
conda run --no-capture-output -n "$ENV" \
  pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# ── smoke test ───────────────────────────────────────────────────────────────
echo "[setup] Running smoke tests"
conda run --no-capture-output -n "$ENV" python - <<'EOF'
import torch, diffusers, transformers
print(f"  torch       {torch.__version__}")
print(f"  diffusers   {diffusers.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  CUDA avail  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "✓  Setup complete."
echo ""
echo "  conda activate $ENV"
echo "  python sd3/gen_images.py --prompts \"a sunset over the ocean\" --out_dir sd3/outputs"
