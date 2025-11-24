#!/usr/bin/env python3
"""
Master Pipeline Orchestration Script

Runs the complete data engineering pipeline:
1. Verify data sources
2. Build crash-level ML dataset
3. Build segment-level ML dataset (optional)

Usage:
    # Full pipeline
    python scripts/run_pipeline.py

    # Sample mode (for testing)
    python scripts/run_pipeline.py --sample 10000

    # Crash-level only
    python scripts/run_pipeline.py --crash-only

Author: Julien Hovan
Date: 2025-11-24
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.paths import ensure_directories


def print_header(text):
    """Print a formatted header"""
    print('\n' + '=' * 80)
    print(text.center(80))
    print('=' * 80 + '\n')


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f'\n>>> {description}')
    print(f'Command: {" ".join(cmd)}')
    print()

    try:
        result = subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        print(f'\n✓ {description} completed successfully')
        return True
    except subprocess.CalledProcessError as e:
        print(f'\n✗ {description} failed with exit code {e.returncode}')
        return False
    except Exception as e:
        print(f'\n✗ {description} failed: {e}')
        return False


def verify_data():
    """Run data verification"""
    print_header('STEP 0: DATA VERIFICATION')
    cmd = [sys.executable, 'scripts/verify_data.py']
    return run_command(cmd, 'Data verification')


def build_crash_dataset(sample_size=None):
    """Build crash-level ML dataset"""
    print_header('STEP 1: BUILD CRASH-LEVEL DATASET')

    cmd = [sys.executable, 'data_engineering/datasets/build_crash_level_dataset.py']

    if sample_size:
        cmd.extend(['--sample', str(sample_size)])

    return run_command(cmd, 'Crash-level dataset builder')


def build_segment_dataset(sample_size=None):
    """Build segment-level ML dataset"""
    print_header('STEP 2: BUILD SEGMENT-LEVEL DATASET')

    cmd = [sys.executable, 'data_engineering/datasets/build_segment_level_dataset.py']

    if sample_size:
        cmd.extend(['--sample', str(sample_size)])

    return run_command(cmd, 'Segment-level dataset builder')


def main():
    """Main pipeline orchestration"""
    parser = argparse.ArgumentParser(
        description='Run the complete data engineering pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/run_pipeline.py

  # Sample mode (for testing)
  python scripts/run_pipeline.py --sample 10000

  # Crash-level only
  python scripts/run_pipeline.py --crash-only

  # Skip verification
  python scripts/run_pipeline.py --skip-verify
        """
    )

    parser.add_argument(
        '--sample',
        type=int,
        help='Use only N samples (for testing)'
    )

    parser.add_argument(
        '--crash-only',
        action='store_true',
        help='Build only crash-level dataset (skip segment-level)'
    )

    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip data verification step'
    )

    args = parser.parse_args()

    # Print banner
    print_header('TEXAS CRASH PREDICTION - DATA PIPELINE')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    if args.sample:
        print(f'Mode: SAMPLE ({args.sample:,} records)')
    else:
        print('Mode: FULL DATASET')

    print()

    # Ensure directories exist
    print('Ensuring directory structure...')
    ensure_directories()
    print('✓ Directory structure ready\n')

    # Track success
    all_success = True
    start_time = datetime.now()

    # Step 0: Verify data (optional)
    if not args.skip_verify:
        if not verify_data():
            print('\n⚠️  Data verification failed. Continue anyway? [y/N] ', end='')
            response = input().strip().lower()
            if response != 'y':
                print('\nPipeline aborted.')
                return 1

    # Step 1: Build crash-level dataset
    if not build_crash_dataset(sample_size=args.sample):
        all_success = False
        print('\n⚠️  Crash-level dataset failed. Continue to segment-level? [y/N] ', end='')
        response = input().strip().lower()
        if response != 'y':
            print('\nPipeline aborted.')
            return 1

    # Step 2: Build segment-level dataset (optional)
    if not args.crash_only:
        if not build_segment_dataset(sample_size=args.sample):
            all_success = False

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print_header('PIPELINE SUMMARY')
    print(f'Started:  {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Finished: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Duration: {duration}')
    print()

    if all_success:
        print('✓ PIPELINE COMPLETED SUCCESSFULLY')
        print()
        print('Next steps:')
        print('  1. Train models: python -m ml_engineering.train_with_mlflow --dataset crash --model all')
        print('  2. Launch app: streamlit run app/Home.py')
        print()
        return 0
    else:
        print('✗ PIPELINE COMPLETED WITH ERRORS')
        print()
        print('Some datasets may not have been created successfully.')
        print('Check the error messages above for details.')
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
