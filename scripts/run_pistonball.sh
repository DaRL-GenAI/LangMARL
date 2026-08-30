#!/bin/bash
# Train on Pistonball. Requires: pip install -e ".[pistonball]"
# Usage: bash scripts/run_pistonball.sh --config configs/pistonball/central_credit.json
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec langmarl train "$@"
