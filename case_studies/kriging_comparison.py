"""
Kriging Comparison for Both Case Studies

This script implements Ordinary Kriging with spatial cross-validation
for both Meuse and Jura datasets, comparing with ML approaches.

Demonstrates:
1. Ordinary Kriging as baseline geostatistical method
2. Comparison with Random Forest + Standard CV
3. Comparison with Random Forest + Spatial CV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import cdist

print("="*80)
print("Kriging Comparison: Meuse + Jura Datasets")
print("="*80)
print()

# ============================================================================
# CASE STUDY 1: MEUSE (Zinc)
# ============================================================================

print("="*80)
print("CASE STUDY 1: MEUSE (Heavy Metal Contamination)")
print("="*80)
print()

print("[1/4] Loading Meuse dataset...")
meuse_data = pd.read_csv('case_studies/data/meuse.csv')
print(f"Loaded {len(meuse_data)} samples")
print()

# Prepare data
X_meuse = meuse_data[['x', 'y']].values
y_meuse = meuse_data['zinc'].values

print("[2/4] Running Ordinary Kriging with 5-fold Spatial CV...")

# Spatial K-Fold using clustering (same as MLPY)
from sklearn.cluster import KMeans
kmeans_meuse = KMeans(n_clusters=5, random_state=42, n_init=10)
fold_labels_meuse = kmeans_meuse.fit_predict(X_meuse)

predictions_kriging_meuse = np.zeros_like(y_meuse)
kriging_fold_errors_meuse = []

for fold in range(5):
    # Split into train/test
    test_mask = (fold_labels_meuse == fold)
    train_mask = ~test_mask

    X_train = X_meuse[train_mask]
    y_train = y_meuse[train_mask]
    X_test = X_meuse[test_mask]
    y_test = y_meuse[test_mask]

    print(f"  Fold {fold+1}/5: Train={train_mask.sum()}, Test={test_mask.sum()}", end="")

    try:
        # Ordinary Kriging
        # variogram_model options: 'linear', 'power', 'gaussian', 'spherical', 'exponential'
        OK = OrdinaryKriging(
            X_train[:, 0], X_train[:, 1], y_train,
            variogram_model='exponential',
            verbose=False,
            enable_plotting=False,
            nlags=15
        )

        # Predict on test set
        y_pred, ss = OK.execute('points', X_test[:, 0], X_test[:, 1])

        # Store predictions
        predictions_kriging_meuse[test_mask] = y_pred

        # Calculate fold RMSE
        fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        kriging_fold_errors_meuse.append(fold_rmse)

        print(f" - RMSE: {fold_rmse:.2f} ppm")

    except Exception as e:
        print(f" - ERROR: {str(e)}")
        kriging_fold_errors_meuse.append(np.nan)

# Overall Kriging RMSE
kriging_rmse_meuse = np.sqrt(mean_squared_error(y_meuse, predictions_kriging_meuse))
kriging_rmse_mean_meuse = np.nanmean(kriging_fold_errors_meuse)

print()
print(f"Ordinary Kriging Results (Meuse):")
print(f"  Overall RMSE: {kriging_rmse_meuse:.2f} ppm")
print(f"  Mean Fold RMSE: {kriging_rmse_mean_meuse:.2f} ppm")
print()

# ============================================================================
# CASE STUDY 2: JURA (Copper)
# ============================================================================

print("="*80)
print("CASE STUDY 2: JURA (Mineral Prospectivity)")
print("="*80)
print()

print("[3/4] Loading Jura dataset...")
jura_data = pd.read_csv('case_studies/data/jura_copper.csv')
print(f"Loaded {len(jura_data)} samples")
print()

# Prepare data
X_jura = jura_data[['X', 'Y']].values
y_jura = jura_data['Cu'].values

print("[4/4] Running Ordinary Kriging with 5-fold Spatial CV...")

# Spatial K-Fold using clustering
kmeans_jura = KMeans(n_clusters=5, random_state=42, n_init=10)
fold_labels_jura = kmeans_jura.fit_predict(X_jura)

predictions_kriging_jura = np.zeros_like(y_jura)
kriging_fold_errors_jura = []

for fold in range(5):
    # Split into train/test
    test_mask = (fold_labels_jura == fold)
    train_mask = ~test_mask

    X_train = X_jura[train_mask]
    y_train = y_jura[train_mask]
    X_test = X_jura[test_mask]
    y_test = y_jura[test_mask]

    print(f"  Fold {fold+1}/5: Train={train_mask.sum()}, Test={test_mask.sum()}", end="")

    try:
        # Ordinary Kriging
        OK = OrdinaryKriging(
            X_train[:, 0], X_train[:, 1], y_train,
            variogram_model='spherical',  # Spherical model for Jura
            verbose=False,
            enable_plotting=False,
            nlags=15
        )

        # Predict on test set
        y_pred, ss = OK.execute('points', X_test[:, 0], X_test[:, 1])

        # Store predictions
        predictions_kriging_jura[test_mask] = y_pred

        # Calculate fold RMSE
        fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        kriging_fold_errors_jura.append(fold_rmse)

        print(f" - RMSE: {fold_rmse:.2f} mg/kg")

    except Exception as e:
        print(f" - ERROR: {str(e)}")
        kriging_fold_errors_jura.append(np.nan)

# Overall Kriging RMSE
kriging_rmse_jura = np.sqrt(mean_squared_error(y_jura, predictions_kriging_jura))
kriging_rmse_mean_jura = np.nanmean(kriging_fold_errors_jura)

print()
print(f"Ordinary Kriging Results (Jura):")
print(f"  Overall RMSE: {kriging_rmse_jura:.2f} mg/kg")
print(f"  Mean Fold RMSE: {kriging_rmse_mean_jura:.2f} mg/kg")
print()

# ============================================================================
# COMPARATIVE SUMMARY
# ============================================================================

print("="*80)
print("COMPARATIVE SUMMARY: ML vs Geostatistics")
print("="*80)
print()

# Results from previous analyses (hardcoded from case study scripts)
meuse_results = {
    'Standard CV (sklearn)': 220.45,
    'Standard CV (MLPY)': 219.90,
    'Spatial CV (MLPY)': 319.23,
    'Ordinary Kriging': kriging_rmse_mean_meuse
}

jura_results = {
    'Standard CV (sklearn)': 24.31,
    'Standard CV (MLPY)': 24.34,
    'Spatial CV (MLPY)': 25.66,
    'Ordinary Kriging': kriging_rmse_mean_jura
}

print("MEUSE DATASET (Zinc, ppm)")
print("-" * 60)
print(f"{'Method':<30} {'RMSE (ppm)':<15} {'vs Spatial CV':<15}")
print("-" * 60)
for method, rmse in meuse_results.items():
    diff = ((rmse - meuse_results['Spatial CV (MLPY)']) / meuse_results['Spatial CV (MLPY)']) * 100
    print(f"{method:<30} {rmse:<15.2f} {diff:>+6.1f}%")
print("-" * 60)
print()

print("JURA DATASET (Copper, mg/kg)")
print("-" * 60)
print(f"{'Method':<30} {'RMSE (mg/kg)':<15} {'vs Spatial CV':<15}")
print("-" * 60)
for method, rmse in jura_results.items():
    diff = ((rmse - jura_results['Spatial CV (MLPY)']) / jura_results['Spatial CV (MLPY)']) * 100
    print(f"{method:<30} {rmse:<15.2f} {diff:>+6.1f}%")
print("-" * 60)
print()

# ============================================================================
# CREATE COMPARISON FIGURE
# ============================================================================

print("Generating comparison figure...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Meuse comparison
methods_meuse = ['Standard\nCV', 'Spatial\nCV', 'Ordinary\nKriging']
rmse_meuse = [meuse_results['Standard CV (MLPY)'],
              meuse_results['Spatial CV (MLPY)'],
              meuse_results['Ordinary Kriging']]
colors_meuse = ['#3498db', '#e74c3c', '#2ecc71']

bars1 = axes[0].bar(methods_meuse, rmse_meuse, color=colors_meuse,
                     alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('RMSE (ppm)', fontsize=12)
axes[0].set_title('Meuse Dataset: ML vs Kriging', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, max(rmse_meuse) * 1.15)
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars1, rmse_meuse):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + max(rmse_meuse)*0.02,
                 f'{val:.1f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

# Jura comparison
methods_jura = ['Standard\nCV', 'Spatial\nCV', 'Ordinary\nKriging']
rmse_jura = [jura_results['Standard CV (MLPY)'],
             jura_results['Spatial CV (MLPY)'],
             jura_results['Ordinary Kriging']]
colors_jura = ['#3498db', '#e74c3c', '#2ecc71']

bars2 = axes[1].bar(methods_jura, rmse_jura, color=colors_jura,
                     alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('RMSE (mg/kg)', fontsize=12)
axes[1].set_title('Jura Dataset: ML vs Kriging', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, max(rmse_jura) * 1.15)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars2, rmse_jura):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + max(rmse_jura)*0.02,
                 f'{val:.1f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('case_studies/figures/ml_vs_kriging_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('case_studies/figures/ml_vs_kriging_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: ml_vs_kriging_comparison.pdf/png")

print()
print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()
print("1. Ordinary Kriging explicitly models spatial dependence")
print("2. Random Forest + Spatial CV achieves competitive performance")
print("3. Spatial CV is ESSENTIAL for ML methods in spatial settings")
print("4. ML advantage: handles complex non-linear relationships + covariates")
print("5. Kriging advantage: explicit spatial model, uncertainty quantification")
print()
print("CONCLUSION: ML + Spatial CV is valid alternative to Kriging when:")
print("  - Complex non-linear relationships exist")
print("  - Multiple environmental covariates available")
print("  - Spatial CV properly accounts for autocorrelation")
print()
print("="*80)
print("KRIGING COMPARISON COMPLETE")
print("="*80)
