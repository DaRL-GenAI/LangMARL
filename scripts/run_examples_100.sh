#!/bin/bash
# Launch the three 100-task language runs in the background, with logs.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting all training jobs at $(date)..."
for task in qa math coding; do
  nohup python "examples/train_${task}.py" > "$LOG_DIR/${task}_${TIMESTAMP}.log" 2>&1 &
  echo "  ${task}  PID=$!  -> $LOG_DIR/${task}_${TIMESTAMP}.log"
done

echo
echo "All jobs launched. Monitor with:"
echo "  tail -f $LOG_DIR/*_${TIMESTAMP}.log"
