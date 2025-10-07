"""
Example of integrating MLPY CLI in Python scripts.

This shows how to call CLI commands programmatically.
"""

import subprocess
import sys
import json
import pandas as pd
from pathlib import Path


def run_mlpy_command(args):
    """Run an MLPY CLI command and return output."""
    # Get the project root directory (parent of examples)
    project_root = Path(__file__).parent.parent
    
    cmd = [sys.executable, "-m", "mlpy"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    return result.stdout


def main():
    print("MLPY CLI Integration Example")
    print("=" * 40)
    
    # 1. Get MLPY info
    print("\n1. Getting MLPY info...")
    info = run_mlpy_command(["info"])
    print(info.split('\n')[1:4])  # Show first few lines
    
    # 2. Train a model and save results
    print("\n2. Training model...")
    output_file = "cli_results.csv"
    train_output = run_mlpy_command([
        "train", "examples/iris_sample.csv",
        "-t", "classif",
        "-y", "species",
        "-l", "rf",
        "-k", "5",
        "-m", "acc",
        "-o", output_file
    ])
    
    if train_output and Path(output_file).exists():
        results = pd.read_csv(output_file)
        print(f"Training complete! Accuracy: {results['mean'].iloc[0]:.3f}")
    
    # 3. Benchmark multiple models
    print("\n3. Benchmarking models...")
    benchmark_output = run_mlpy_command([
        "benchmark", "examples/iris_sample.csv",
        "-t", "classif",
        "-y", "species",
        "-l", "rf", "-l", "lr", "-l", "dt",
        "-k", "3"
    ])
    
    # Parse benchmark results
    if benchmark_output:
        lines = benchmark_output.strip().split('\n')
        print("Benchmark Results:")
        # Find and print the ranking section
        for i, line in enumerate(lines):
            if "Ranking:" in line:
                for j in range(i, min(i+5, len(lines))):
                    print(lines[j])
                break
    
    # 4. Create experiment configuration programmatically
    print("\n4. Creating experiment configuration...")
    experiment_config = {
        "name": "Automated Iris Experiment",
        "description": "Created via CLI integration",
        "data": {
            "file": "examples/iris_sample.csv",
            "target": "species",
            "task_type": "classif"
        },
        "learners": [
            {"type": "rf", "params": {"n_estimators": 100, "random_state": 42}},
            {"type": "lr", "params": {"max_iter": 1000}},
            {"type": "dt", "params": {"max_depth": 5}}
        ],
        "resampling": {
            "method": "cv",
            "folds": 5
        },
        "measures": ["acc", "f1"],
        "output": {
            "results": "automated_experiment_results.csv",
            "plots": False
        }
    }
    
    with open("automated_experiment.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    print("Experiment configuration saved to automated_experiment.json")
    
    # Clean up
    for file in [output_file, "automated_experiment.json"]:
        if Path(file).exists():
            Path(file).unlink()
    
    print("\n[OK] CLI integration example complete!")


if __name__ == "__main__":
    main()