#!/bin/bash
# Train on a language task (QA / Math / Coding / Writing).
# Usage: bash scripts/run_language_task.sh --config configs/language_task/qa_central_credit.json [OPTIONS]
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec langmarl train "$@"
