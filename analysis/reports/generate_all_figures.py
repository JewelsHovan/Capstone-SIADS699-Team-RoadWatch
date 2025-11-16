#!/usr/bin/env python3
"""
Generate All Report Figures

Convenience script to generate all 5 figures for the capstone report in sequence.

Usage:
    python analysis/reports/generate_all_figures.py

    # Or with specific figures only:
    python analysis/reports/generate_all_figures.py --figures 1 2 3

Author: Capstone Team
Date: 2025-11-15
"""

import sys
import argparse
from pathlib import Path
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_figure_script(figure_num, script_name):
    """Run a figure generation script"""
    print('\n' + '='*80)
    print(f'GENERATING FIGURE {figure_num}: {script_name}')
    print('='*80)

    script_path = Path(__file__).parent / f'figure_{figure_num}_{script_name}.py'

    if not script_path.exists():
        print(f'❌ Script not found: {script_path}')
        return False

    try:
        start_time = time.time()

        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print('STDERR:', result.stderr, file=sys.stderr)

        if result.returncode == 0:
            print(f'\n✅ Figure {figure_num} complete! ({elapsed:.1f}s)')
            return True
        else:
            print(f'\n❌ Figure {figure_num} failed with exit code {result.returncode}')
            return False

    except subprocess.TimeoutExpired:
        print(f'\n⏱️  Figure {figure_num} timed out after 10 minutes')
        return False
    except Exception as e:
        print(f'\n❌ Error running Figure {figure_num}: {e}')
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Generate all figures for capstone report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all figures
  python generate_all_figures.py

  # Generate specific figures only
  python generate_all_figures.py --figures 1 2 3

  # Skip Figure 5 (requires manual screenshots)
  python generate_all_figures.py --figures 1 2 3 4
        """
    )

    parser.add_argument(
        '--figures',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        help='Specific figures to generate (default: all)'
    )

    parser.add_argument(
        '--skip-5',
        action='store_true',
        help='Skip Figure 5 (app screenshots) - same as --figures 1 2 3 4'
    )

    args = parser.parse_args()

    # Handle --skip-5 flag
    if args.skip_5:
        figures_to_generate = [1, 2, 3, 4]
    else:
        figures_to_generate = args.figures

    print('='*80)
    print('CAPSTONE REPORT FIGURES GENERATION')
    print('='*80)
    print(f'\nGenerating figures: {", ".join(map(str, figures_to_generate))}')
    print(f'Project root: {project_root}')

    # Create output directory
    output_dir = project_root / 'analysis' / 'reports' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_dir}')

    # Figure names
    figure_scripts = {
        1: 'geospatial',
        2: 'feature_importance',
        3: 'model_comparison',
        4: 'regularization_comparison',
        5: 'app_screenshots'
    }

    # Track results
    results = {}
    start_time_total = time.time()

    # Generate each figure
    for fig_num in figures_to_generate:
        if fig_num in figure_scripts:
            success = run_figure_script(fig_num, figure_scripts[fig_num])
            results[fig_num] = success
        else:
            print(f'\n⚠️  Unknown figure number: {fig_num}')
            results[fig_num] = False

    total_elapsed = time.time() - start_time_total

    # Summary
    print('\n' + '='*80)
    print('GENERATION SUMMARY')
    print('='*80)

    for fig_num in figures_to_generate:
        status = '✅ Success' if results.get(fig_num, False) else '❌ Failed'
        print(f'  Figure {fig_num} ({figure_scripts[fig_num]}): {status}')

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    print(f'\nTotal: {success_count}/{total_count} figures generated successfully')
    print(f'Time elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)')

    print(f'\n📁 Outputs saved to: {output_dir}/')

    # Special note for Figure 5
    if 5 in figures_to_generate:
        print('\n📸 Note for Figure 5 (App Screenshots):')
        print('   - Requires manual screenshot capture from Streamlit app')
        print('   - See figure_5_app_screenshots.py for detailed instructions')
        print('   - A placeholder has been created as a reference')

    # Exit code based on success
    if success_count == total_count:
        print('\n🎉 All figures generated successfully!')
        sys.exit(0)
    else:
        print(f'\n⚠️  {total_count - success_count} figure(s) failed')
        sys.exit(1)

if __name__ == '__main__':
    main()
