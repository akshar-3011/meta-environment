#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-command multi-agent training pipeline
#
# Usage:
#   bash examples/run_training.sh                    # Train all 3 archetypes
#   bash examples/run_training.sh --wandb            # With W&B logging
#   bash examples/run_training.sh --timesteps 100000 # Custom timesteps
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  🏋️  Meta-Environment Multi-Agent Training Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Ensure dependencies ─────────────────────────────────────────────
echo "📦 Checking dependencies..."
if command -v uv &>/dev/null; then
    uv sync --dev 2>/dev/null || true
    uv pip install stable-baselines3 gymnasium tensorboard pyyaml 2>/dev/null || true
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    pip install -q stable-baselines3 gymnasium tensorboard pyyaml 2>/dev/null || true
else
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q -e ".[dev]"
    pip install -q stable-baselines3 gymnasium tensorboard pyyaml
fi
echo "  ✅ Dependencies ready"
echo ""

# ── Step 2: Run training ────────────────────────────────────────────────────
echo "🏋️ Starting multi-agent training..."
echo "  Configs: training/configs/{conservative,aggressive,balanced}.yaml"
echo ""

python training/train_all.py \
    --config training/configs/ \
    --save-dir models/ \
    --checkpoint-freq 10000 \
    "$@"

echo ""

# ── Step 3: Compare agents ──────────────────────────────────────────────────
echo "🔍 Running agent comparison..."
python training/compare_agents.py \
    --models-dir models/ \
    --episodes 5 \
    --output-dir results/

echo ""

# ── Step 4: Summary ─────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ Pipeline complete!"
echo ""
echo "  📁 Models:       models/{conservative,aggressive,balanced}/"
echo "  📊 Results:      results/comparison_*.csv"
echo "  📋 Details:      results/comparison_*.json"
echo ""
echo "  Next steps:"
echo "    tensorboard --logdir models/    # View training curves"
echo "    python training/compare_agents.py --episodes 20  # More thorough eval"
echo "═══════════════════════════════════════════════════════════════"
