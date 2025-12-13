"""
Replication Scripts for MLPY JSS Paper

This script reproduces all figures and tables from:
"MLPY: A Modular Machine Learning Framework for Python with Native Spatial Support"
Francisco Parra O. and Giuseppe Cavallieri

Journal of Statistical Software (2025)

Requirements:
- mlpy-geo >= 0.1.0
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.13.0
- matplotlib >= 3.3.0

Usage:
    python replicate_all.py

Outputs:
    All figures and tables will be saved to ./output/
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import mlpy
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

# Import MLPY components
try:
    from mlpy.tasks import TaskRegrSpatial
    from mlpy.learners.sklearn.regression import LearnerRandomForestRegressor
    from mlpy.resamplings import ResamplingCV, SpatialKFold
    from mlpy.measures import MeasureRegrRMSE
    from mlpy import resample
    print("[OK] MLPY imported successfully")
except ImportError as e:
    print(f"[ERROR] Error importing MLPY: {e}")
    print("Please install mlpy-geo: pip install mlpy-geo")
    sys.exit(1)

# Import scipy for spatial statistics
try:
    from scipy.io import loadmat
    from scipy.spatial.distance import cdist
    print("[OK] scipy imported successfully")
except ImportError:
    print("[ERROR] Error importing scipy")
    print("Please install scipy: pip install scipy")
    sys.exit(1)

# Set random seeds for reproducibility
np.random.seed(42)

# =============================================================================
# Helper Functions
# =============================================================================

def load_meuse_data():
    """
    Load REAL Meuse river dataset.

    The Meuse dataset contains heavy metal contamination measurements
    from the Meuse river floodplain in the Netherlands.

    Returns:
        pd.DataFrame: Meuse dataset with columns:
            - x, y: coordinates
            - zinc, cadmium, lead, copper: contamination (ppm)
            - dist: distance to river
            - elev: elevation
    """
    try:
        # Load REAL Meuse data from case_studies
        meuse_path = Path(__file__).parent.parent / 'case_studies' / 'data' / 'meuse.csv'

        if not meuse_path.exists():
            raise FileNotFoundError(
                f"Meuse dataset not found at {meuse_path}. "
                "Please ensure case_studies/data/meuse.csv exists."
            )

        print(f"Loading REAL Meuse dataset from {meuse_path.name}...")
        df = pd.read_csv(meuse_path)

        # Verify required columns exist
        required_cols = ['x', 'y', 'zinc', 'cadmium', 'lead', 'copper', 'dist', 'elev']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"[OK] Loaded Meuse dataset: {len(df)} observations (REAL DATA)")
        print(f"     Columns: {', '.join(df.columns.tolist()[:8])}")  # Show first 8 columns
        return df

    except Exception as e:
        print(f"[ERROR] Error loading Meuse data: {e}")
        sys.exit(1)


def calculate_morans_i(y, coords, weights='inverse_distance'):
    """
    Calculate Moran's I statistic for spatial autocorrelation.

    Args:
        y: np.array of values
        coords: np.array of coordinates (n x 2)
        weights: weight type ('inverse_distance' or 'binary')

    Returns:
        float: Moran's I statistic
    """
    n = len(y)
    y_mean = np.mean(y)
    y_dev = y - y_mean

    # Calculate spatial weights matrix
    distances = cdist(coords, coords)

    if weights == 'inverse_distance':
        # Inverse distance weights (avoid division by zero)
        W = 1.0 / (distances + 1e-6)
        np.fill_diagonal(W, 0)  # No self-weight
    else:
        # Binary weights (within threshold)
        threshold = np.median(distances)
        W = (distances < threshold).astype(float)
        np.fill_diagonal(W, 0)

    # Moran's I formula
    numerator = n * np.sum(W * np.outer(y_dev, y_dev))
    denominator = np.sum(W) * np.sum(y_dev**2)

    I = numerator / denominator
    return I


def create_variogram(df, var='zinc', n_bins=15):
    """
    Create empirical variogram for visualization.

    Args:
        df: DataFrame with spatial data
        var: variable name
        n_bins: number of distance bins

    Returns:
        tuple: (distances, semivariances)
    """
    coords = df[['x', 'y']].values
    values = df[var].values

    # Calculate all pairwise distances and squared differences
    distances = cdist(coords, coords)
    sq_diff = (values[:, None] - values[None, :])**2

    # Flatten and remove diagonal
    dist_flat = distances[np.triu_indices_from(distances, k=1)]
    sq_diff_flat = sq_diff[np.triu_indices_from(sq_diff, k=1)]

    # Bin by distance
    bins = np.linspace(0, dist_flat.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    semivariances = []
    for i in range(len(bins) - 1):
        mask = (dist_flat >= bins[i]) & (dist_flat < bins[i + 1])
        if mask.sum() > 0:
            semivariances.append(0.5 * sq_diff_flat[mask].mean())
        else:
            semivariances.append(np.nan)

    return bin_centers, semivariances


# =============================================================================
# Figure 1: Spatial Distribution of Zinc
# =============================================================================

def replicate_figure1(df, output_dir='output'):
    """
    Figure 1: Spatial distribution of zinc concentration in Meuse floodplain.
    """
    print("\n" + "="*60)
    print("Replicating Figure 1: Spatial Distribution")
    print("="*60)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with color scale
    scatter = ax.scatter(
        df['x'], df['y'], c=df['zinc'],
        cmap='YlOrRd', s=100, alpha=0.7,
        edgecolors='black', linewidths=0.5
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Zinc Concentration (ppm)')

    # Labels
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title('Spatial Distribution of Zinc Contamination\\nMeuse River Floodplain',
                 fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Save
    output_path = Path(output_dir) / 'figure1_spatial_distribution.pdf'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


# =============================================================================
# Figure 2: Variogram
# =============================================================================

def replicate_figure2(df, output_dir='output'):
    """
    Figure 2: Empirical variogram for zinc concentration.
    """
    print("\n" + "="*60)
    print("Replicating Figure 2: Variogram")
    print("="*60)

    # Calculate variogram
    distances, semivariances = create_variogram(df, var='zinc', n_bins=15)

    # Remove NaN values
    mask = ~np.isnan(semivariances)
    distances = distances[mask]
    semivariances = np.array(semivariances)[mask]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot empirical variogram
    ax.scatter(distances, semivariances, s=100, alpha=0.6,
              edgecolors='black', linewidths=1, label='Empirical')

    # Polynomial fit (trend line)
    z = np.polyfit(distances, semivariances, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(distances.min(), distances.max(), 100)
    ax.plot(x_smooth, p(x_smooth), 'r--', linewidth=2, label='Polynomial Fit')

    # Labels
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Semivariance', fontsize=12)
    ax.set_title('Empirical Variogram for Zinc Concentration',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')

    # Save
    output_path = Path(output_dir) / 'figure2_variogram.pdf'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


# =============================================================================
# Table 1: Moran's I Statistics
# =============================================================================

def replicate_table1(df, output_dir='output'):
    """
    Table 1: Moran's I statistics for heavy metals.
    """
    print("\n" + "="*60)
    print("Replicating Table 1: Moran's I Statistics")
    print("="*60)

    coords = df[['x', 'y']].values
    metals = ['zinc', 'cadmium', 'lead', 'copper']

    results = []
    for metal in metals:
        I = calculate_morans_i(df[metal].values, coords)

        # Interpretation
        if I > 0.25:
            interp = "Moderate"
        elif I > 0.10:
            interp = "Weak"
        else:
            interp = "None"

        results.append({
            'Metal': metal.capitalize(),
            'Morans_I': f"{I:.3f}",
            'Interpretation': interp
        })
        print(f"  {metal.capitalize():10s}: I = {I:.3f} ({interp})")

    # Save as CSV
    results_df = pd.DataFrame(results)
    output_path = Path(output_dir) / 'table1_morans_i.csv'
    results_df.to_csv(output_path, index=False)
    print(f"[OK] Saved: {output_path}")

    return results_df


# =============================================================================
# Table 2: Cross-Validation Comparison
# =============================================================================

def replicate_table2_and_figures(df, output_dir='output'):
    """
    Table 2: CV method comparison + Figures 3 & 4.
    """
    print("\n" + "="*60)
    print("Replicating Table 2: CV Comparison")
    print("="*60)

    # Prepare features and target
    X = df[['dist', 'elev']].values
    y = df['zinc'].values
    coords = df[['x', 'y']].values

    # Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Method 1: sklearn regular CV
    print("\n1. sklearn regular CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_sklearn = -cross_val_score(rf, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    rmse_sklearn = scores_sklearn.mean()
    print(f"   RMSE: {rmse_sklearn:.1f} ppm")

    # Method 2: MLPY regular CV
    print("\n2. MLPY regular CV...")
    # TaskRegrSpatial auto-selects features (all columns except target and coordinates)
    task = TaskRegrSpatial(
        data=df[['x', 'y', 'zinc', 'dist', 'elev']],
        target='zinc',
        coordinate_names=['x', 'y']
    )
    learner = LearnerRandomForestRegressor(n_estimators=100, random_state=42)
    resampling_regular = ResamplingCV(folds=5)
    measure = MeasureRegrRMSE()

    result_regular = resample(task, learner, resampling_regular, [measure])
    rmse_mlpy_regular = result_regular.score('regr.rmse')
    print(f"   RMSE: {rmse_mlpy_regular:.1f} ppm")

    # Method 3: MLPY Spatial K-Fold
    print("\n3. MLPY Spatial K-Fold...")
    resampling_spatial = SpatialKFold(n_folds=5)
    result_spatial = resample(task, learner, resampling_spatial, [measure])
    rmse_mlpy_spatial = result_spatial.score('regr.rmse')
    print(f"   RMSE: {rmse_mlpy_spatial:.1f} ppm")

    # Calculate underestimation
    underest_sklearn = ((rmse_mlpy_spatial - rmse_sklearn) / rmse_mlpy_spatial) * 100
    underest_mlpy = ((rmse_mlpy_spatial - rmse_mlpy_regular) / rmse_mlpy_spatial) * 100

    print(f"\n[OK] Regular CV underestimates error by ~{underest_sklearn:.1f}%")

    # Save table
    results = pd.DataFrame({
        'Method': [
            'sklearn CV',
            'MLPY standard CV',
            'MLPY Spatial K-Fold'
        ],
        'RMSE_ppm': [
            f"{rmse_sklearn:.1f}",
            f"{rmse_mlpy_regular:.1f}",
            f"{rmse_mlpy_spatial:.1f}"
        ],
        'Underestimation': [
            f"{underest_sklearn:.1f}%",
            f"{underest_mlpy:.1f}%",
            "---"
        ]
    })

    output_path = Path(output_dir) / 'table2_cv_comparison.csv'
    results.to_csv(output_path, index=False)
    print(f"[OK] Saved: {output_path}")

    # Figure 3: Bar plot comparison
    print("\nGenerating Figure 3: CV Comparison Bar Plot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['sklearn\nCV', 'MLPY\nRegular CV', 'MLPY\nSpatial CV']
    rmses = [rmse_sklearn, rmse_mlpy_regular, rmse_mlpy_spatial]
    colors = ['#1f77b4', '#1f77b4', '#ff7f0e']  # Blue for regular, orange for spatial

    bars = ax.bar(methods, rmses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('RMSE (ppm)', fontsize=12)
    ax.set_title('Cross-Validation Method Comparison\\nZinc Concentration Prediction',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(rmses) * 1.15)

    output_path = Path(output_dir) / 'figure3_cv_comparison.pdf'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()

    # Figure 4: Spatial folds visualization
    print("\nGenerating Figure 4: Spatial Folds...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Get fold assignments from spatial resampling
    spatial_resampling_viz = SpatialKFold(n_folds=5)
    spatial_resampling_viz.instantiate(task)

    # Get data subset (matching task data)
    task_data = df[['x', 'y', 'zinc', 'dist', 'elev']].reset_index(drop=True)

    for fold_idx in range(5):
        ax = axes[fold_idx]

        # Get train/test indices for this fold
        train_idx = spatial_resampling_viz.train_set(fold_idx)
        test_idx = spatial_resampling_viz.test_set(fold_idx)

        # Plot train (blue) and test (red)
        ax.scatter(task_data.iloc[list(train_idx)]['x'], task_data.iloc[list(train_idx)]['y'],
                  c='blue', s=50, alpha=0.6, label='Train')
        ax.scatter(task_data.iloc[list(test_idx)]['x'], task_data.iloc[list(test_idx)]['y'],
                  c='red', s=50, alpha=0.8, label='Test')

        ax.set_title(f'Fold {fold_idx + 1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide extra subplot
    axes[5].axis('off')

    fig.suptitle('Spatial K-Fold Cross-Validation (5 Folds)\\nGeographic Separation of Train/Test Sets',
                 fontsize=16, fontweight='bold')

    output_path = Path(output_dir) / 'figure4_spatial_folds.pdf'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()

    return results


# =============================================================================
# Session Info (Reproducibility)
# =============================================================================

def print_session_info():
    """
    Print Python environment information (equivalent to R's sessionInfo()).

    This function reports the Python version, platform, and package versions
    to ensure reproducibility of results as required by JSS.
    """
    import platform

    print("\n" + "="*70)
    print("SESSION INFO")
    print("="*70)

    # Python version
    print(f"\nPython version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    # Package versions
    print("\nPackage versions:")
    packages = [
        'mlpy',
        'numpy',
        'pandas',
        'sklearn',
        'scipy',
        'matplotlib'
    ]

    for pkg_name in packages:
        try:
            # Handle sklearn special case (package is 'sklearn' but import is 'sklearn')
            if pkg_name == 'sklearn':
                import sklearn
                mod = sklearn
            else:
                mod = __import__(pkg_name)

            version = getattr(mod, '__version__', 'unknown')
            print(f"  {pkg_name:15s} {version}")
        except ImportError:
            print(f"  {pkg_name:15s} not installed")

    print("\n" + "="*70)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main function to replicate all results.
    """
    print("="*70)
    print("MLPY JSS Paper - Replication Script")
    print("="*70)
    print("\nThis script reproduces all figures and tables from the paper.")
    print("Outputs will be saved to: ./output/\n")

    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    print(f"[OK] Output directory: {output_dir.absolute()}\n")

    # Load data
    df = load_meuse_data()

    # Replicate all figures and tables
    replicate_figure1(df, output_dir)
    replicate_figure2(df, output_dir)
    replicate_table1(df, output_dir)
    replicate_table2_and_figures(df, output_dir)

    print("\n" + "="*70)
    print("[OK] Replication Complete!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - figure1_spatial_distribution.pdf")
    print("  - figure2_variogram.pdf")
    print("  - figure3_cv_comparison.pdf")
    print("  - figure4_spatial_folds.pdf")
    print("  - table1_morans_i.csv")
    print("  - table2_cv_comparison.csv")
    print("\nThese files can be compared with the paper's figures and tables.")
    print("="*70)

    # Print session info for reproducibility (JSS requirement)
    print_session_info()


if __name__ == "__main__":
    main()
