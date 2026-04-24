#!/bin/bash

# Do not exit immediately if a command fails, so we can continue to the next experiment.
set +e

echo "Starting IR experiments and evaluations..."

# Change to the directory containing run.py to ensure relative paths work correctly
cd "$(dirname "$0")"

for config in configs/*.yaml; do
    echo "=================================================="
    echo "Running configuration: $config"
    echo "=================================================="
    
    # Extract run_name from the yaml file to use in the metrics step
    # This handles both double quotes and no quotes (fallback)
    run_name=$(grep -E '^run_name:' "$config" | awk -F'"' '{print $2}')
    
    if [ -z "$run_name" ]; then
        run_name=$(grep -E '^run_name:' "$config" | awk '{print $2}' | tr -d '"'\''')
    fi
    
    # Run generation if not already done
    if [ ! -f "runs/$run_name/generations.jsonl" ]; then
        if ! python run.py generate --config "$config"; then
            echo "Error: Generation failed for $config. Skipping to next configuration."
            continue
        fi
    else
        echo "Generation for run $run_name already exists. Skipping generation."
    fi
    
    #check if evaluation already done
    if [ -f "runs/$run_name/metrics.json" ]; then
        echo "Evaluation for run $run_name already exists. Skipping."
        continue
    fi
    
    if [ -n "$run_name" ]; then
        echo "--------------------------------------------------"
        echo "Evaluating run: runs/$run_name"
        echo "--------------------------------------------------"
        # Run metrics evaluation
        if ! python run.py metrics --run "runs/$run_name"; then
            echo "Error: Evaluation failed for run $run_name."
            continue
        fi
    else
        echo "Warning: Could not extract run_name from $config. Skipping evaluation for this config."
    fi
    
    echo ""
done

echo "All experiments and evaluations completed successfully!"
