#!/usr/bin/env bash
# Run the whole Leaf Lens pipeline in order and stop at the first failure.
#
#   ./scripts/run_pipeline.sh                      # default config (v1)
#   ./scripts/run_pipeline.sh config/model_v2.yaml # a variant
#
# Requires images in data/raw/<Class_Name>/. Every step prints where its
# output went; nothing is skipped silently.

set -euo pipefail

CONFIG_ARG=()
if [[ $# -ge 1 ]]; then
  CONFIG_ARG=(--config "$1")
  echo "Using configuration: $1"
fi

PY=${PYTHON:-python}
cd "$(dirname "$0")/.."

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

step "1/6  Validating data/raw"
"$PY" -m src.data.validate_dataset "${CONFIG_ARG[@]}" --quiet

step "2/6  Preparing train / validation / test splits"
"$PY" -m src.data.prepare_dataset "${CONFIG_ARG[@]}"

step "3/6  Checking the splits for data leakage"
"$PY" -m src.data.validate_dataset "${CONFIG_ARG[@]}" --split-report

step "4/6  Training"
"$PY" -m src.model.train "${CONFIG_ARG[@]}"

step "5/6  Evaluating on the unseen test split"
"$PY" -m src.model.evaluate "${CONFIG_ARG[@]}" --quiet

step "6/6  Analysing errors"
"$PY" -m src.testing.error_analysis "${CONFIG_ARG[@]}" --quiet

printf '\n\033[1mPipeline complete.\033[0m\n'
echo "  Model:            models/<version>/model.keras"
echo "  Evaluation:       results/evaluation/"
echo "  Confusion matrix: results/confusion_matrices/"
echo "  Error analysis:   results/error_analysis/"
echo "  Graphs:           results/graphs/"
echo
echo "Next:"
echo "  python -m src.testing.realistic_testing --template   # field-condition testing"
echo "  python app/app.py                                    # the web interface"
