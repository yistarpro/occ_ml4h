#!/bin/bash
# The whole audit of one model, step by step.
#   bash run_audit.sh merlin | renalclip | medsiglip | validation   [output folder, default results/<model>]
set -euo pipefail
cd "$(dirname "$0")"
M=${1:?model: merlin | renalclip | medsiglip | validation}
O=${2:-results/$M}; mkdir -p "$O"

audit() {   # model, output folder, extra flag
  python run.py "$1" drops     --out "$2/drops.csv" $3
  python run.py "$1" guard_fit --out "$2/guard.npz" $3
  python run.py "$1" guard     --guard-fit "$2/guard.npz" --drops "$2/drops.csv" --out "$2/guard.csv" $3
}

case $M in
  merlin|renalclip|medsiglip)
    [ "$M" = merlin ] && python scripts/totalsegmentator_masks.py
    audit "$M" "$O" ""
    python run.py "$M" position --guard-fit "$O/guard.npz" --out "$O/position.json"
    DET=""
    if [ "$M" = medsiglip ]; then python run.py medsiglip detection --out "$O/detection.csv"; DET="--detection $O/detection.csv"; fi
    python summarize.py --drops "$O/drops.csv" --guard "$O/guard.csv" --position "$O/position.json" $DET --out "$O/summary.json" ;;
  validation)
    mkdir -p "$O/randomized_encoder" "$O/grounded_scorer"
    audit merlin "$O/randomized_encoder" --random-encoder
    python summarize.py --drops "$O/randomized_encoder/drops.csv" --guard "$O/randomized_encoder/guard.csv" --out "$O/randomized_encoder/summary.json"
    python run.py merlin_grounded drops --out "$O/grounded_scorer/drops.csv"
    python summarize.py --drops "$O/grounded_scorer/drops.csv" --out "$O/grounded_scorer/summary.json" ;;
  *) echo "unknown model $M"; exit 1 ;;
esac
