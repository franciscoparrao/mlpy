"""
Master Benchmark Script

Runs all benchmarks and generates a comprehensive report for the JMLR paper.

Run with: python benchmarks/run_all_benchmarks.py
"""

import sys
import time
import traceback

# Set up path
sys.path.insert(0, '.')

print("=" * 80)
print("MLPY JMLR Benchmark Suite")
print("=" * 80)
print()
print("This script will run all benchmarks required for the JMLR-MLOSS paper:")
print("  1. Performance Benchmark (MLPY vs sklearn)")
print("  2. Usability Benchmark (code complexity)")
print("  3. Spatial ML Benchmark (unique capability)")
print()
print("Estimated time: 5-10 minutes")
print("=" * 80)
print()
print("Starting benchmarks...")
print()

results = {}

# ============================================================================
# 1. Performance Benchmark
# ============================================================================
print("\n" + "=" * 80)
print("1/3: Running Performance Benchmark...")
print("=" * 80)
print()

try:
    start_time = time.time()
    from benchmarks.performance_benchmark import run_benchmarks
    df_performance = run_benchmarks()
    elapsed = time.time() - start_time
    results['performance'] = {
        'status': 'SUCCESS',
        'time': elapsed,
        'data': df_performance
    }
    print(f"\n✓ Performance benchmark completed in {elapsed:.1f}s")
except Exception as e:
    print(f"\n✗ Performance benchmark failed: {e}")
    traceback.print_exc()
    results['performance'] = {'status': 'FAILED', 'error': str(e)}

# ============================================================================
# 2. Usability Benchmark
# ============================================================================
print("\n" + "=" * 80)
print("2/3: Running Usability Benchmark...")
print("=" * 80)
print()

try:
    start_time = time.time()
    from benchmarks.usability_benchmark import analyze_usability
    df_usability = analyze_usability()
    elapsed = time.time() - start_time
    results['usability'] = {
        'status': 'SUCCESS',
        'time': elapsed,
        'data': df_usability
    }
    print(f"\n✓ Usability benchmark completed in {elapsed:.1f}s")
except Exception as e:
    print(f"\n✗ Usability benchmark failed: {e}")
    traceback.print_exc()
    results['usability'] = {'status': 'FAILED', 'error': str(e)}

# ============================================================================
# 3. Spatial ML Benchmark
# ============================================================================
print("\n" + "=" * 80)
print("3/3: Running Spatial ML Benchmark...")
print("=" * 80)
print()

try:
    start_time = time.time()
    from benchmarks.spatial_benchmark import benchmark_spatial_cv
    df_spatial = benchmark_spatial_cv()
    elapsed = time.time() - start_time
    results['spatial'] = {
        'status': 'SUCCESS',
        'time': elapsed,
        'data': df_spatial
    }
    print(f"\n✓ Spatial benchmark completed in {elapsed:.1f}s")
except Exception as e:
    print(f"\n✗ Spatial benchmark failed: {e}")
    traceback.print_exc()
    results['spatial'] = {'status': 'FAILED', 'error': str(e)}

# ============================================================================
# Final Report
# ============================================================================
print("\n\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)
print()

success_count = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
total_count = len(results)

print(f"Benchmarks completed: {success_count}/{total_count}")
print()

for name, result in results.items():
    status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
    status_text = "SUCCESS" if result['status'] == 'SUCCESS' else "FAILED"
    print(f"{status_symbol} {name.capitalize()}: {status_text}", end='')
    if result['status'] == 'SUCCESS':
        print(f" ({result['time']:.1f}s)")
    else:
        print(f" - {result.get('error', 'Unknown error')}")

print()
print("=" * 80)
print("Output files:")
print("  - benchmarks/results/performance_results.csv")
print("  - benchmarks/results/usability_results.csv")
print("  - benchmarks/results/spatial_results.csv")
print("=" * 80)
print()

if success_count == total_count:
    print("✓ All benchmarks completed successfully!")
    print()
    print("Next steps for JMLR paper:")
    print("  1. Review the CSV files in benchmarks/results/")
    print("  2. Create plots using the data (matplotlib/seaborn)")
    print("  3. Include results in the paper's Benchmark section")
    print("  4. Discuss implications in the Discussion section")
else:
    print("⚠ Some benchmarks failed. Please review the errors above.")

print()
print("=" * 80)
