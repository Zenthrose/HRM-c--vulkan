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
import argparse
from pathlib import Path


def run_curriculum_learning(root_dir, curriculum_stages, start_stage, max_stage):
    """Run curriculum learning through progressive stages"""
    print("🌱 Starting Curriculum Learning Journey")
    print(f"Total stages: {len(curriculum_stages)}")
    print(f"Target: Stage {max_stage} ({curriculum_stages[max_stage]['name']})")
    print()

    # Track progress
    curriculum_progress = []

    for stage_idx in range(start_stage, min(max_stage + 1, len(curriculum_stages))):
        stage = curriculum_stages[stage_idx]

        print(f"🎯 Stage {stage_idx}: {stage['name']}")
        print(f"   Goal: {stage['description']}")
        print(f"   Context: {stage['context_length']} chars")
        print(
            f"   Data: {stage['data_percentage'] * 100:.1f}% ({int(stage['data_percentage'] * 319501)} sequences)"
        )
        print(
            f"   Batch: {stage['batch_size']}, LR: {stage['learning_rate']}, Epochs: {stage['epochs']}"
        )
        print()

        # Create stage-specific config
        stage_config = create_stage_config(root_dir, stage, stage_idx)

        # Run training for this stage
        success = run_stage_training(root_dir, stage_config, stage_idx)

        if success:
            curriculum_progress.append(
                {
                    "stage": stage_idx,
                    "name": stage["name"],
                    "config": stage_config,
                    "completed": True,
                }
            )
            print(f"✅ Stage {stage_idx} completed successfully!")
        else:
            print(f"❌ Stage {stage_idx} failed - curriculum paused")
            break

        print("\n" + "=" * 60 + "\n")

    # Save curriculum progress
    progress_file = root_dir / "logs" / "curriculum_progress.json"
    progress_file.parent.mkdir(exist_ok=True)
    with open(progress_file, "w") as f:
        json.dump(curriculum_progress, f, indent=2)

    print("📊 Curriculum Learning Complete!")
    print(f"Progress saved to: {progress_file}")


def create_stage_config(root_dir, stage, stage_idx):
    """Create configuration for a specific curriculum stage"""
    config_path = root_dir / "config" / "character_training_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Override with stage-specific parameters
    config["training"]["context_length"] = stage["context_length"]
    config["training"]["batch_size"] = stage["batch_size"]
    config["training"]["learning_rate"] = stage["learning_rate"]
    config["training"]["max_epochs"] = stage["epochs"]
    config["data"]["data_percentage"] = stage["data_percentage"]

    # Scale model parameters for early curriculum stages to prevent memory issues
    if stage_idx == 0:  # Foundation stage
        config["model"]["char_vocab_size"] = 1000  # Reduced from 100k
        config["model"]["hidden_size"] = 128       # Reduced from 768
        config["model"]["num_layers"] = 2          # Reduced from 12
        config["model"]["num_heads"] = 4           # Reduced from 12
    elif stage_idx == 1:  # Pattern Recognition stage
        config["model"]["char_vocab_size"] = 5000  # Reduced from 100k
        config["model"]["hidden_size"] = 256       # Reduced from 768
        config["model"]["num_layers"] = 4          # Reduced from 12
        config["model"]["num_heads"] = 8           # Reduced from 12
    # Later stages use full model parameters from base config

    # Save stage config
    stage_config_path = (
        root_dir
        / "config"
        / f"stage_{stage['name'].lower().replace(' ', '_')}_config.json"
    )
    with open(stage_config_path, "w") as f:
        json.dump(config, f, indent=2)

    return str(stage_config_path)


def run_stage_training(root_dir, stage_config_path, stage_idx):
    """Run training for a specific stage"""
    exe_path = root_dir / "build" / "release" / "nyx_system"
    if not exe_path.exists():
        print(f"❌ Nyx executable not found: {exe_path}")
        return False

    try:
        # Run training with stage-specific config
        cmd = [str(exe_path), "--train", "config", stage_config_path]
        print(f"Running: {' '.join(cmd)}")
        print(f"Using stage-specific config: {stage_config_path}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            print("Training completed successfully")
            return True
        else:
            print(f"Training failed with return code: {result.returncode}")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False

    except subprocess.TimeoutExpired:
        print("Training timed out after 1 hour")
        return False
    except Exception as e:
        print(f"Training error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="HRM Character-Level Language Training with Curriculum Learning"
    )
    parser.add_argument(
        "--root-dir", type=str, default=".", help="Root directory for HRM project"
    )
    parser.add_argument(
        "--curriculum", action="store_true", help="Enable curriculum learning mode"
    )
    parser.add_argument(
        "--start-stage", type=int, default=0, help="Starting curriculum stage (0-6)"
    )
    parser.add_argument(
        "--max-stage", type=int, default=6, help="Maximum curriculum stage to reach"
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    # Curriculum learning stages (optimized for Intel i7, 16GB RAM, Iris Xe iGPU)
    curriculum_stages = [
        {
            "name": "Foundation",
            "context_length": 128,     # Further reduced to 128 for integrated GPU
            "data_percentage": 0.005,  # Reduced to 0.5% (1.6k sequences)
            "batch_size": 1,
            "learning_rate": 0.001,
            "epochs": 1,               # Reduced to 1 epoch for initial training
            "description": "Basic patterns and vocabulary",
        },
        {
            "name": "Pattern Recognition",
            "context_length": 1024,
            "data_percentage": 0.05,  # 5% of dataset (~16k sequences)
            "batch_size": 2,
            "learning_rate": 0.0005,
            "epochs": 5,
            "description": "Common phrases and structures",
        },
        {
            "name": "Context Expansion",
            "context_length": 2048,
            "data_percentage": 0.10,  # 10% of dataset (~32k sequences)
            "batch_size": 2,
            "learning_rate": 0.0002,
            "epochs": 7,
            "description": "Multi-sentence understanding",
        },
        {
            "name": "Reasoning Development",
            "context_length": 4096,
            "data_percentage": 0.20,  # 20% of dataset (~64k sequences)
            "batch_size": 2,
            "learning_rate": 0.0001,
            "epochs": 8,
            "description": "Logical and scientific reasoning",
        },
        {
            "name": "Deep Integration",
            "context_length": 8192,
            "data_percentage": 0.40,  # 40% of dataset (~128k sequences)
            "batch_size": 2,
            "learning_rate": 0.00005,
            "epochs": 10,
            "description": "Cross-domain knowledge synthesis",
        },
        {
            "name": "Advanced Mastery",
            "context_length": 16384,
            "data_percentage": 0.70,  # 70% of dataset (~224k sequences)
            "batch_size": 2,
            "learning_rate": 0.00002,
            "epochs": 12,
            "description": "Complex academic and creative tasks",
        },
        {
            "name": "Ultimate Context",
            "context_length": 32768,
            "data_percentage": 1.0,  # 100% of dataset (~319k sequences)
            "batch_size": 1,
            "learning_rate": 0.00001,
            "epochs": 15,
            "description": "100k+ context capability with full knowledge",
        },
    ]

    if args.curriculum:
        print("🧠 Nyx Curriculum Learning Mode")
        print("=" * 40)
        print(
            f"Starting at stage {args.start_stage}, aiming for stage {args.max_stage}"
        )
        print(f"Hardware: Intel i7, 16GB RAM, Intel Iris Xe iGPU")
        print()

        # Run curriculum learning
        run_curriculum_learning(
            root_dir, curriculum_stages, args.start_stage, args.max_stage
        )
        return

    print("HRM Character-Level Language Training")
    print("=" * 40)

    # Check for config
    config_path = root_dir / "config" / "character_training_config.json"
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        print("Run: ./prepare_language_dataset.sh")
        return

    print("Config loaded")

    # Check for dataset
    dataset_path = (
        root_dir / "data" / "text" / "processed" / "comprehensive_training_corpus.txt"
    )
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Run: ./prepare_language_dataset.sh")
        return

    print("Dataset found")

    # Check for built executable
    exe_base = "build/release/nyx_system"
    exe_path = exe_base + (".exe" if os.name == "nt" else "")
    if not os.path.exists(exe_path):
        print(f"Nyx executable not found: {exe_path}")
        print("Run: Build the project first using the instructions in AGENTS.md")
        return

    print("Nyx executable found")

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
            universal_newlines=True,
        )

        # Monitor output in real-time
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # Get return code
        return_code = process.poll()

        if return_code == 0:
            print("\nCharacter-level training completed successfully!")
            print("HRM is now trained on character-level language patterns!")

            # Check for training results
            results_file = root_dir / "logs" / "character_training_stats.json"
            if os.path.exists(results_file):
                print(f"Training results saved to: {results_file}")
                try:
                    with open(results_file, "r") as f:
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
        if "process" in locals():
            process.terminate()
            # Resource-aware timeout calculation
            try:
                import psutil

                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent

                # Calculate adaptive timeout based on system resources
                base_timeout = 30  # 30 seconds base
                if cpu_usage > 80 or memory_usage > 80:
                    base_timeout *= 2  # Double under high load
                elif cpu_usage > 60 or memory_usage > 60:
                    base_timeout = int(
                        base_timeout * 1.5
                    )  # 50% increase under medium load
                elif cpu_usage < 20 and memory_usage < 20:
                    base_timeout = int(base_timeout * 0.7)  # Reduce under low load

                process.wait(timeout=base_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
    except Exception as e:
        print(f"\nError during training: {e}")


if __name__ == "__main__":
    main()
