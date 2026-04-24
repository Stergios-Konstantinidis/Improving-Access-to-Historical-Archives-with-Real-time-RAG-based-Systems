#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting BM25 evaluations..."

# Change to the directory containing run.py to ensure relative paths work correctly
cd "$(dirname "$0")"

# Iterate over all BM25 run directories
for run_dir in runs/*; do
    echo "Processing $run_dir..."

    if [ -f "$run_dir/summary.json" ]; then
        echo "Summary already exists. Skipping..."
        continue
    fi
    if [ -d "$run_dir" ]; then
        echo "=================================================="
        echo "Evaluating run: $run_dir"
        echo "=================================================="
        
        # Run metrics evaluation
        python run.py metrics --run "$run_dir"
        
        echo ""
    fi
done

echo "All BM25 evaluations completed successfully!"
