#!/usr/bin/env python
"""
Test runner script for MLPY with coverage reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run MLPY tests with coverage')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--coverage', action='store_true', default=True, help='Generate coverage report')
    parser.add_argument('--html', action='store_true', help='Generate HTML coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--markers', '-m', help='Run tests with specific markers')
    parser.add_argument('--parallel', '-n', help='Run tests in parallel (number of workers)')
    parser.add_argument('--failfast', '-x', action='store_true', help='Stop on first failure')
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd_parts = ['pytest']
    
    # Test selection
    if args.unit:
        cmd_parts.append('tests/unit')
    elif args.integration:
        cmd_parts.append('tests/integration')
    else:
        cmd_parts.append('tests/')
    
    # Verbosity
    if args.verbose:
        cmd_parts.append('-vv')
    else:
        cmd_parts.append('-v')
    
    # Coverage
    if args.coverage:
        cmd_parts.extend([
            '--cov=mlpy',
            '--cov-report=term-missing',
            '--cov-report=xml',
            '--cov-fail-under=90'
        ])
        
        if args.html:
            cmd_parts.append('--cov-report=html')
    
    # Markers
    if args.markers:
        cmd_parts.extend(['-m', args.markers])
    
    # Parallel execution
    if args.parallel:
        cmd_parts.extend(['-n', args.parallel])
    
    # Fail fast
    if args.failfast:
        cmd_parts.append('-x')
    
    # Add color output
    cmd_parts.append('--color=yes')
    
    # Run tests
    test_cmd = ' '.join(cmd_parts)
    success = run_command(test_cmd, "Running Tests")
    
    if not success:
        print("\n❌ Tests failed!")
        sys.exit(1)
    
    # Show coverage report location
    if args.coverage and args.html:
        print(f"\n📊 HTML coverage report generated at: htmlcov/index.html")
        print("   Open in browser: python -m http.server 8000 --directory htmlcov")
    
    print(f"\n✅ All tests passed!")
    
    # Check coverage threshold
    if args.coverage:
        print("\n📈 Coverage Summary:")
        subprocess.run("coverage report", shell=True)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())