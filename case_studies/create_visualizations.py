"""
Create Professional Visualizations for Meuse Spatial ML Case Study

Generates publication-quality figures (300 DPI, PDF) for JMLR paper:
1. Spatial distribution map of zinc concentrations
2. Variogram showing spatial autocorrelation
3. CV methods comparison boxplot
4. Spatial folds visualization

All figures saved to case_studies/plots/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist

# MLPY imports
import sys
sys.path.insert(0, '..')
from mlpy.tasks import TaskRegrSpatial
from mlpy.learners.sklearn import LearnerRandomForestRegressor
from mlpy.measures import MeasureRegrRMSE
from mlpy.resamplings import ResamplingCV

# Import spatial resampling
try:
    from mlpy.resamplings.spatial import SpatialKFold
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    print("Warning: Spatial resampling not available")

from get_meuse_dataset import load_meuse_dataset

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Professional color palette
COLORS = {
    'high': '#d73027',  # Red for high contamination
    'medium': '#fee08b',  # Yellow
    'low': '#1a9850',  # Green for low contamination
    'regular': '#4575b4',  # Blue
    'spatial': '#d73027',  # Red
    'sklearn': '#91bfdb',  # Light blue
}


def create_output_directory():
    """Create output directory for plots."""
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    return output_dir


def figure1_spatial_distribution(df, output_dir):
    """
    Figure 1: Spatial Distribution Map of Zinc Concentrations

    Shows the geographic distribution of zinc contamination with
    color-coded concentrations.
    """
    print("Creating Figure 1: Spatial Distribution Map...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create scatter plot with zinc concentrations
    scatter = ax.scatter(
        df['x'],
        df['y'],
        c=df['zinc'],
        cmap='RdYlGn_r',  # Red (high) to Green (low)
        s=80,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Zinc Concentration (ppm)')

    # Labels and title
    ax.set_xlabel('X Coordinate (meters)')
    ax.set_ylabel('Y Coordinate (meters)')
    ax.set_title('Spatial Distribution of Zinc Contamination\nMeuse River Floodplain (N=155)')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add annotation
    ax.text(
        0.02, 0.98,
        f"Min: {df['zinc'].min():.0f} ppm\nMax: {df['zinc'].max():.0f} ppm\nMean: {df['zinc'].mean():.0f} ppm",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=8
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / 'figure1_spatial_distribution.pdf'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    # Also save PNG for preview
    output_file_png = output_dir / 'figure1_spatial_distribution.png'
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_file_png}")

    plt.close()


def calculate_variogram(coords, values, n_bins=20):
    """
    Calculate empirical variogram.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        Spatial coordinates
    values : array-like, shape (n,)
        Values at each location
    n_bins : int
        Number of distance bins

    Returns
    -------
    distances : array
        Binned distances
    semivariances : array
        Semivariance for each distance bin
    """
    # Calculate pairwise distances
    dists = cdist(coords, coords)

    # Calculate squared differences
    n = len(values)
    diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            diff_matrix[i, j] = (values[i] - values[j]) ** 2
            diff_matrix[j, i] = diff_matrix[i, j]

    # Flatten and remove diagonal
    dists_flat = dists[np.triu_indices(n, k=1)]
    diffs_flat = diff_matrix[np.triu_indices(n, k=1)]

    # Bin distances
    max_dist = dists_flat.max()
    bins = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate semivariance for each bin
    semivariances = []
    for i in range(len(bins) - 1):
        mask = (dists_flat >= bins[i]) & (dists_flat < bins[i+1])
        if mask.sum() > 0:
            semivariances.append(diffs_flat[mask].mean() / 2)
        else:
            semivariances.append(np.nan)

    return bin_centers, np.array(semivariances)


def figure2_variogram(df, output_dir):
    """
    Figure 2: Variogram Showing Spatial Autocorrelation

    Shows how spatial correlation decreases with distance.
    """
    print("Creating Figure 2: Variogram...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate variogram
    coords = df[['x', 'y']].values
    values = df['zinc'].values

    distances, semivariances = calculate_variogram(coords, values, n_bins=15)

    # Remove NaN values
    valid = ~np.isnan(semivariances)
    distances = distances[valid]
    semivariances = semivariances[valid]

    # Plot
    ax.scatter(distances, semivariances, s=80, alpha=0.6,
               color=COLORS['high'], edgecolors='black', linewidth=0.5)

    # Add trend line
    if len(distances) > 3:
        z = np.polyfit(distances, semivariances, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(distances.min(), distances.max(), 100)
        ax.plot(x_smooth, p(x_smooth), '--', color='gray', alpha=0.8,
                linewidth=2, label='Trend')

    # Labels and title
    ax.set_xlabel('Distance (meters)')
    ax.set_ylabel('Semivariance (ppm²)')
    ax.set_title('Spatial Variogram for Zinc Concentration\nShowing Spatial Autocorrelation Structure')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()

    # Add annotation explaining autocorrelation
    ax.text(
        0.98, 0.02,
        "Increasing semivariance with distance\nindicates spatial autocorrelation",
        transform=ax.transAxes,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=8
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / 'figure2_variogram.pdf'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    output_file_png = output_dir / 'figure2_variogram.png'
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_file_png}")

    plt.close()


def figure3_cv_comparison(output_dir):
    """
    Figure 3: Cross-Validation Methods Comparison

    Barplot comparing RMSE across three CV methods, showing
    overestimation by regular CV.
    """
    print("Creating Figure 3: CV Methods Comparison...")

    # Load results
    import json
    results_file = Path('results') / 'meuse_analysis_results.json'
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract RMSE values
    cv_results = results['cv_comparison']
    methods = []
    rmse_values = []
    colors_list = []

    for key, data in cv_results.items():
        if data:
            methods.append(data['method'])
            rmse_values.append(data['rmse'])
            if 'Spatial' in data['method']:
                colors_list.append(COLORS['spatial'])
            elif 'sklearn' in data['method']:
                colors_list.append(COLORS['sklearn'])
            else:
                colors_list.append(COLORS['regular'])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create bar plot
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, rmse_values, color=colors_list, alpha=0.7,
                   edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, rmse) in enumerate(zip(bars, rmse_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add overestimation annotation
    if len(rmse_values) >= 3:
        regular_rmse = rmse_values[1]  # MLPY Regular
        spatial_rmse = rmse_values[2]  # MLPY Spatial
        overest = cv_results['mlpy_spatial_cv']['overestimation_pct']

        # Draw arrow showing difference
        ax.annotate('', xy=(1, spatial_rmse), xytext=(1, regular_rmse),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))

        # Add text annotation
        mid_y = (regular_rmse + spatial_rmse) / 2
        ax.text(1.3, mid_y, f'{overest:.1f}%\nunderestimation',
                fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Labels and title
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('RMSE (ppm)')
    ax.set_title('Cross-Validation Method Comparison\nRegular CV Underestimates True Error')

    # Grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_file = output_dir / 'figure3_cv_comparison.pdf'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    output_file_png = output_dir / 'figure3_cv_comparison.png'
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_file_png}")

    plt.close()


def figure4_spatial_folds(df, output_dir):
    """
    Figure 4: Spatial Folds Visualization

    Shows how SpatialKFold divides data spatially to prevent leakage.
    """
    print("Creating Figure 4: Spatial Folds Visualization...")

    if not SPATIAL_AVAILABLE:
        print("  Skipping: Spatial resampling not available")
        return

    # Prepare data
    feature_cols = ['dist', 'elev', 'cadmium', 'copper']
    target = 'zinc'
    df_clean = df.dropna(subset=feature_cols + [target])
    df_spatial = df_clean[['x', 'y'] + feature_cols + [target]].copy()

    # Create spatial task
    task = TaskRegrSpatial(
        data=df_spatial,
        target=target,
        coordinate_names=['x', 'y']
    )

    # Get spatial folds
    spatial_resampling = SpatialKFold(n_folds=5, random_state=42)

    # Instantiate resampling with task
    spatial_resampling.instantiate(task)

    # Get coordinates
    coords = df_spatial[['x', 'y']].values

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # Plot all data in first subplot
    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], s=50, alpha=0.5,
               color='gray', edgecolors='black', linewidth=0.5)
    ax.set_title('All Data (N=155)', fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)

    # Plot each fold
    fold_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for fold_idx in range(5):
        train_idx = spatial_resampling.train_set(fold_idx)
        test_idx = spatial_resampling.test_set(fold_idx)
        ax = axes[fold_idx + 1]

        # Plot train points
        ax.scatter(coords[train_idx, 0], coords[train_idx, 1],
                   s=30, alpha=0.3, color='gray', label='Train')

        # Plot test points
        ax.scatter(coords[test_idx, 0], coords[test_idx, 1],
                   s=80, alpha=0.8, color=fold_colors[fold_idx],
                   edgecolors='black', linewidth=0.8, label='Test')

        ax.set_title(f'Fold {fold_idx + 1} (Test: {len(test_idx)} samples)',
                     fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Spatial K-Fold Cross-Validation\nTest Sets Are Spatially Separated',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    output_file = output_dir / 'figure4_spatial_folds.pdf'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    output_file_png = output_dir / 'figure4_spatial_folds.png'
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_file_png}")

    plt.close()


def create_all_figures():
    """Create all publication-quality figures."""
    print("=" * 80)
    print("CREATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Load dataset
    print("Loading Meuse dataset...")
    df = load_meuse_dataset()
    print(f"Dataset loaded: {len(df)} observations")
    print()

    # Create figures
    figure1_spatial_distribution(df, output_dir)
    print()

    figure2_variogram(df, output_dir)
    print()

    figure3_cv_comparison(output_dir)
    print()

    figure4_spatial_folds(df, output_dir)
    print()

    print("=" * 80)
    print("ALL FIGURES CREATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print(f"Location: {output_dir.absolute()}")
    print()
    print("Files generated:")
    print("  - figure1_spatial_distribution.pdf/png")
    print("  - figure2_variogram.pdf/png")
    print("  - figure3_cv_comparison.pdf/png")
    print("  - figure4_spatial_folds.pdf/png")
    print()
    print("These figures are ready for inclusion in the JMLR paper.")
    print("Resolution: 300 DPI")
    print("Format: PDF (vector) + PNG (preview)")


if __name__ == '__main__':
    import time
    start = time.time()

    create_all_figures()

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")
