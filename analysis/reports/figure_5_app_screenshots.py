#!/usr/bin/env python3
"""
Figure 5: Deployed Crash Severity Prediction System (Streamlit App)

This script provides instructions and utilities for capturing screenshots
from the Streamlit application for the final report.

The figure should showcase:
- Panel A: Segment Risk Map (geographic risk visualization)
- Panel B: Input form for crash severity prediction
- Panel C: Prediction output with risk assessment

Author: Capstone Team
Date: 2025-11-15
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path('analysis/reports/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*80)
print('FIGURE 5: DEPLOYED CRASH SEVERITY PREDICTION SYSTEM')
print('='*80)

# ============================================================================
# INSTRUCTIONS FOR CAPTURING SCREENSHOTS
# ============================================================================
print('\n📸 SCREENSHOT CAPTURE INSTRUCTIONS:')
print('='*80)

print("""
STEP 1: Start the Streamlit App
--------------------------------
Run the following command in your terminal:
    cd /Users/julienh/Desktop/MADS/Capstone
    streamlit run app/Home.py

STEP 2: Navigate and Capture Screenshots
-----------------------------------------
Capture the following screenshots and save them in:
    analysis/reports/figures/screenshots/

Required Screenshots:

1. screenshot_segment_risk_map.png
   - Navigate to "Segment Risk Map" page
   - Zoom into a interesting area (e.g., Houston or Dallas metro)
   - Show color-coded road segments (green/yellow/red)
   - Include legend and map controls
   - Recommended size: Full window, ~1920x1080

2. screenshot_predictor_input.png
   - Navigate to "Crash Severity Predictor" page
   - Show the input form with sliders/dropdowns
   - Fill in some example values:
     * AADT: 50,000
     * Speed Limit: 65 mph
     * Weather: Clear
     * Time: 5:00 PM (17:00)
     * Number of Lanes: 4
   - Capture before clicking "Predict"
   - Focus on the input section

3. screenshot_predictor_output.png
   - After filling inputs, click "Predict"
   - Capture the prediction output:
     * Risk probability (e.g., 72%)
     * Risk level (High/Medium/Low)
     * Contributing factors
   - Include any visualizations (gauge, bar chart)

STEP 3: Arrange Screenshots
----------------------------
Once you have the screenshots, uncomment and run the code below
to create a composite figure for the report.

TIPS:
- Use Chrome/Firefox in full screen for best quality
- Hide browser toolbars (F11 for full screen)
- Ensure good contrast and readability
- Take high-resolution screenshots (Retina/HiDPI if available)
- You can use browser dev tools to set consistent viewport size
""")

print('\n' + '='*80)

# ============================================================================
# Create Screenshot Directory
# ============================================================================
SCREENSHOT_DIR = OUTPUT_DIR / 'screenshots'
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
print(f'\n✅ Screenshot directory created: {SCREENSHOT_DIR}/')

# ============================================================================
# Create Composite Figure (run after screenshots are captured)
# ============================================================================
def create_composite_figure():
    """
    Creates a composite figure from captured screenshots.
    Run this function after you've captured all required screenshots.
    """

    screenshot_files = {
        'map': SCREENSHOT_DIR / 'screenshot_segment_risk_map.png',
        'input': SCREENSHOT_DIR / 'screenshot_predictor_input.png',
        'output': SCREENSHOT_DIR / 'screenshot_predictor_output.png'
    }

    # Check if all screenshots exist
    missing = [name for name, path in screenshot_files.items() if not path.exists()]

    if missing:
        print(f'\n⚠️  Missing screenshots: {", ".join(missing)}')
        print('\nPlease capture the required screenshots first.')
        print('See instructions above.')
        return False

    print('\n📸 All screenshots found! Creating composite figure...')

    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], width_ratios=[1.2, 1],
                  hspace=0.15, wspace=0.15)

    # Panel A: Segment Risk Map (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    img1 = mpimg.imread(str(screenshot_files['map']))
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('Panel A: Segment Risk Map - Geographic Risk Visualization',
                 fontsize=14, fontweight='bold', pad=15, loc='left')

    # Panel B: Input Form (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    img2 = mpimg.imread(str(screenshot_files['input']))
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('Panel B: Prediction Input Interface',
                 fontsize=13, fontweight='bold', pad=10, loc='left')

    # Panel C: Prediction Output (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    img3 = mpimg.imread(str(screenshot_files['output']))
    ax3.imshow(img3)
    ax3.axis('off')
    ax3.set_title('Panel C: Risk Assessment Output',
                 fontsize=13, fontweight='bold', pad=10, loc='left')

    # Overall title
    fig.suptitle('Deployed Streamlit Application: Real-Time Crash Severity Prediction for Texas Roadways',
                fontsize=16, fontweight='bold', y=0.98)

    # Add caption/takeaway
    caption = ('Our deployed application enables DOT planners to assess crash risk in real-time by inputting '
              'work zone characteristics (traffic volume, speed limit, weather, time) and receiving immediate '
              'risk predictions for informed decision-making on work zone scheduling and safety resource allocation.')

    fig.text(0.5, 0.02, caption, ha='center', fontsize=11, style='italic', wrap=True,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Save
    output_path = OUTPUT_DIR / 'figure_5_app_deployment.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: {output_path}')
    plt.close()

    print('\n✅ Figure 5 complete!')
    return True

# ============================================================================
# Create Placeholder/Template Figure
# ============================================================================
def create_placeholder_figure():
    """
    Creates a placeholder figure showing the expected layout.
    Use this to plan your screenshots.
    """

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], width_ratios=[1.2, 1],
                  hspace=0.15, wspace=0.15)

    # Panel A
    ax1 = fig.add_subplot(gs[0, :])
    ax1.text(0.5, 0.5, 'Panel A\n\nSegment Risk Map\n\nInteractive map with color-coded road segments\n(green=low, yellow=medium, red=high risk)',
            ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Panel A: Segment Risk Map - Geographic Risk Visualization',
                 fontsize=14, fontweight='bold', pad=15, loc='left')

    # Panel B
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.text(0.5, 0.5, 'Panel B\n\nInput Form\n\nSliders/dropdowns for:\n- AADT (traffic volume)\n- Speed limit\n- Weather conditions\n- Time of day\n- Number of lanes',
            ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Panel B: Prediction Input Interface',
                 fontsize=13, fontweight='bold', pad=10, loc='left')

    # Panel C
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.text(0.5, 0.5, 'Panel C\n\nPrediction Output\n\n- Risk probability (0-100%)\n- Risk level classification\n- Contributing factors\n- Visualization (gauge/bar)',
            ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Panel C: Risk Assessment Output',
                 fontsize=13, fontweight='bold', pad=10, loc='left')

    fig.suptitle('Figure 5: Deployed Streamlit Application (PLACEHOLDER - Replace with Screenshots)',
                fontsize=16, fontweight='bold', y=0.98)

    output_path = OUTPUT_DIR / 'figure_5_placeholder.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'  ✓ Created placeholder: {output_path}')
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    # Create placeholder to show expected layout
    print('\nCreating placeholder figure...')
    create_placeholder_figure()

    print('\n' + '='*80)
    print('NEXT STEPS:')
    print('='*80)
    print(f'\n1. Capture screenshots following the instructions above')
    print(f'2. Save screenshots to: {SCREENSHOT_DIR}/')
    print(f'3. Run this script again with screenshots in place')
    print(f'\nOr run the create_composite_figure() function directly:')
    print(f'    python -c "from figure_5_app_screenshots import create_composite_figure; create_composite_figure()"')

    # Try to create composite (will fail gracefully if screenshots don't exist)
    print('\nAttempting to create composite figure...')
    success = create_composite_figure()

    if not success:
        print('\n💡 TIP: Use the placeholder as a reference for composing your screenshots!')
        print(f'   View: {OUTPUT_DIR / "figure_5_placeholder.png"}')
