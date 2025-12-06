#!/usr/bin/env python3
"""
HRM Character-Level Language Training Script
"""

import json
import os
import sys
import time
import subprocess
import signal
from pathlib import Path

def main():
    print("HRM Character-Level Language Training")
    print("=" * 40)

    # Check for config
    config_path = "config/character_training_config.json"
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        print("Run: ./prepare_language_dataset.sh")
        return

    print("Config loaded")

    # Check for dataset
    dataset_path = "data/text/processed/comprehensive_training_corpus.txt"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Run: ./prepare_language_dataset.sh")
        return

    print("Dataset found")

    # Check for built executable
    exe_path = "build/release/hrm_system.exe"
    if not os.path.exists(exe_path):
        print(f"HRM executable not found: {exe_path}")
        print("Run: Build the project first using the instructions in AGENTS.md")
        return

    print("HRM executable found")

    # Run actual Vulkan-based training
    print("\nStarting Character-Level Training (Vulkan)")
    print("=" * 50)

    try:
        # Run the training command
        cmd = [exe_path, "--train"]
        print(f"Running: {' '.join(cmd)}")
        print(f"Using executable: {exe_path}")

        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Monitor output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # Get return code
        return_code = process.poll()

        if return_code == 0:
            print("\nCharacter-level training completed successfully!")
            print("HRM is now trained on character-level language patterns!")

            # Check for training results
            results_file = "logs/character_training_stats.json"
            if os.path.exists(results_file):
                print(f"Training results saved to: {results_file}")
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    print("Final metrics:")
                    for key, value in results.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                except:
                    print("  (Could not parse results file)")

        else:
            print(f"\nTraining failed with return code: {return_code}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if 'process' in locals():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    except Exception as e:
        print(f"\nError during training: {e}")

if __name__ == "__main__":
    main()