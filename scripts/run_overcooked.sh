#!/bin/bash
# Train on Overcooked. Requires: pip install -e ".[overcooked]"
# Usage: bash scripts/run_overcooked.sh --config configs/overcooked/cramped_room_central_credit.json
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec langmarl train "$@"
