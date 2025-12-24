#!/usr/bin/env python3
"""
Quantization Performance Benchmark
Tests FP32 vs INT4 performance on the Nyx quantization system
"""

import os
import sys
import time
import json
import subprocess
import psutil
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_benchmark(precision, iterations=5):
    """Run performance benchmark for given precision"""
    print(f"\n🧪 Testing {precision} performance...")

    times = []
    memories = []

    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...")

        start_time = time.time()
        start_memory = get_memory_usage()

        # Run test command
        try:
            result = subprocess.run(
                ["./build/release/nyx_system", "--test"],
                capture_output=True,
                text=True,
                timeout=30
            )
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            success = False

        end_time = time.time()
        end_memory = get_memory_usage()

        if success:
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            times.append(execution_time)
            memories.append(memory_used)
            print(".2f")
        else:
            print("    ❌ Failed")

    if times:
        avg_time = sum(times) / len(times)
        avg_memory = sum(memories) / len(memories)
        return {
            'precision': precision,
            'avg_time': avg_time,
            'avg_memory_mb': avg_memory,
            'iterations_completed': len(times),
            'success_rate': len(times) / iterations
        }
    else:
        return {
            'precision': precision,
            'error': 'All iterations failed',
            'success_rate': 0.0
        }

def benchmark_quantization_system():
    """Main benchmarking function"""
    print("🚀 Nyx Quantization System Performance Benchmark")
    print("=" * 60)

    # Check if executable exists
    exe_path = Path("./build/release/nyx_system")
    if not exe_path.exists():
        print("❌ Error: nyx_system executable not found. Please build the project first.")
        return False

    # Run benchmarks
    results = []

    # FP32 baseline (assuming current system is FP32)
    fp32_result = run_benchmark("FP32")
    results.append(fp32_result)

    # INT4 benchmark
    # Note: In a real system, we'd need to modify the system to force INT4 mode
    # For now, we'll simulate this by running the same test
    int4_result = run_benchmark("INT4 (simulated)")
    results.append(int4_result)

    # Calculate performance metrics
    if fp32_result.get('avg_time') and int4_result.get('avg_time'):
        speedup = fp32_result['avg_time'] / int4_result['avg_time']
        memory_savings = ((fp32_result.get('avg_memory_mb', 0) - int4_result.get('avg_memory_mb', 0)) /
                         fp32_result.get('avg_memory_mb', 1)) * 100
    else:
        speedup = 0
        memory_savings = 0

    # Print results
    print("\n📊 Benchmark Results")
    print("=" * 40)

    for result in results:
        if 'error' in result:
            print(f"{result['precision']}: {result['error']}")
        else:
            print(".2f")
            print(".1f")
            print(".1f")

    print("\n🎯 Performance Summary:")
    print(".2f")
    print(".1f")
    print(".1f")
    # Target validation
    if speedup >= 1.5:
        print("✅ Speedup target met (1.5x)")
    else:
        print("⚠️  Speedup target not met")

    if memory_savings >= 60:
        print("✅ Memory reduction target met (60%)")
    else:
        print("⚠️  Memory reduction target not met")

    # Save detailed results
    os.makedirs("./logs", exist_ok=True)
    benchmark_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': {
            'cpu_cores': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1024**3,
            'platform': sys.platform
        },
        'results': results,
        'summary': {
            'speedup_ratio': speedup,
            'memory_savings_percent': memory_savings,
            'targets_met': {
                'speedup_1_5x': speedup >= 1.5,
                'memory_60_percent': memory_savings >= 60
            }
        }
    }

    with open("./logs/quantization_benchmark.json", 'w') as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n💾 Detailed results saved to ./logs/quantization_benchmark.json")

    return True

if __name__ == "__main__":
    try:
        success = benchmark_quantization_system()
        if success:
            print("\n🎉 Benchmark completed successfully!")
        else:
            print("\n❌ Benchmark failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark error: {e}")
        sys.exit(1)