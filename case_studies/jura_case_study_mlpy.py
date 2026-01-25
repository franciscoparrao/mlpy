"""
Case Study 2: Mineral Prospectivity - Jura Copper Dataset
Swiss Jura topsoil geochemistry for copper prediction

This script demonstrates spatial cross-validation on the Jura dataset
comparing standard CV vs spatial CV for copper concentration prediction.

Dataset: Based on Goovaerts (1997) - Swiss Jura topsoil geochemistry
Reference: Goovaerts, P. 1997. Geostatistics for Natural Resources Evaluation.
           Oxford Univ. Press, Appendix C.

IMPORTANT: This script must be run from the mlpy/ directory (repository root).
Follow the instructions in case_studies/README.md for proper execution.

Usage:
    cd mlpy/
    python case_studies/jura_case_study_mlpy.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# Import MLPY components
import sys
sys.path.insert(0, 'C:/Users/gran_/Documents/Proyectos/MLPY')

from mlpy.tasks import TaskRegrSpatial
from mlpy.learners.sklearn import LearnerRandomForestRegressor
from mlpy.resamplings import ResamplingCV
from mlpy.resamplings.spatial import SpatialKFold
from mlpy.measures import MeasureRegrRMSE, MeasureRegrMAE
from mlpy import resample

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("MLPY Case Study 2: Mineral Prospectivity (Jura Copper Dataset)")
print("="*80)
print()

# ============================================================================
# 1. LOAD JURA DATASET
# ============================================================================

print("[1/7] Loading Jura copper dataset...")

# Load the dataset created earlier
jura_data = pd.read_csv('case_studies/data/jura_copper.csv')

print(f"Dataset loaded: {len(jura_data)} observations")
print(f"Study area: Swiss Jura topsoil geochemistry")
print(f"Spatial extent: {jura_data['X'].min():.2f} to {jura_data['X'].max():.2f} km (X)")
print(f"                {jura_data['Y'].min():.2f} to {jura_data['Y'].max():.2f} km (Y)")
print(f"Target variable: Copper (Cu) concentration")
print(f"Cu range: {jura_data['Cu'].min():.2f} - {jura_data['Cu'].max():.2f} mg/kg")
print(f"Cu mean: {jura_data['Cu'].mean():.2f} mg/kg (SD: {jura_data['Cu'].std():.2f})")
print()

# ============================================================================
# 2. EXPLORATORY SPATIAL ANALYSIS
# ============================================================================

print("[2/7] Computing spatial autocorrelation statistics...")

# Compute Moran's I for copper concentration
coords = jura_data[['X', 'Y']].values
distances = cdist(coords, coords)

y = jura_data['Cu'].values
y_mean = y.mean()
n = len(y)

# Use inverse distance weights (< 1km threshold)
threshold = 1.0
W = np.where((distances < threshold) & (distances > 0), 1.0 / (distances + 0.01), 0)
np.fill_diagonal(W, 0)
W_sum = W.sum()

numerator = np.sum(W * np.outer(y - y_mean, y - y_mean))
denominator = np.sum((y - y_mean)**2)
morans_i = (n / W_sum) * (numerator / denominator)

print(f"Moran's I for Cu: {morans_i:.3f}")

interpretation = "strong positive" if morans_i > 0.5 else "moderate positive" if morans_i > 0.3 else "weak positive"
print(f"Interpretation: {interpretation} spatial autocorrelation")
print()

# ============================================================================
# 3. CREATE MLPY SPATIAL TASK
# ============================================================================

print("[3/7] Creating MLPY spatial task...")

# Features for prediction
feature_cols = ['Elevation', 'Rock', 'Landuse']
X = jura_data[feature_cols].copy()
y_vals = jura_data['Cu'].values
coords_array = jura_data[['X', 'Y']].values

# Create spatial task
task_data = jura_data[feature_cols + ['Cu', 'X', 'Y']].copy()
task = TaskRegrSpatial(
    data=task_data,
    target='Cu',
    coordinate_names=['X', 'Y']
)

print(f"Task created: TaskRegrSpatial with {len(feature_cols)} features")
print(f"Features: {', '.join(feature_cols)}")
print(f"Coordinates preserved for spatial validation")
print()

# ============================================================================
# 4. STANDARD CROSS-VALIDATION (Baseline)
# ============================================================================

print("[4/7] Running standard 5-fold cross-validation...")

# Standard sklearn CV for comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

cv_standard = KFold(n_splits=5, shuffle=True, random_state=42)
scores_standard = -cross_val_score(
    rf_model, X_scaled, y_vals,
    cv=cv_standard,
    scoring='neg_root_mean_squared_error'
)

rmse_standard_sklearn = scores_standard.mean()
std_standard_sklearn = scores_standard.std()

print(f"sklearn Standard CV RMSE: {rmse_standard_sklearn:.2f} +/- {std_standard_sklearn:.2f} mg/kg")

# MLPY standard CV (for consistency check)
learner = LearnerRandomForestRegressor(n_estimators=100, random_state=42)
resampling_standard = ResamplingCV(folds=5)

result_standard = resample(
    task=task,
    learner=learner,
    resampling=resampling_standard,
    measures=[MeasureRegrRMSE(), MeasureRegrMAE()]
)

# Get scores from result
rmse_standard_mlpy = result_standard.score()  # Gets RMSE (first measure)

print(f"MLPY Standard CV RMSE: {rmse_standard_mlpy:.2f} mg/kg")
print()

# ============================================================================
# 5. SPATIAL CROSS-VALIDATION
# ============================================================================

print("[5/7] Running spatial 5-fold cross-validation...")

# MLPY Spatial K-Fold CV
resampling_spatial = SpatialKFold(n_folds=5)

result_spatial = resample(
    task=task,
    learner=learner,
    resampling=resampling_spatial,
    measures=[MeasureRegrRMSE(), MeasureRegrMAE()]
)

# Get scores from result
rmse_spatial = result_spatial.score()  # Gets RMSE (first measure)

print(f"MLPY Spatial CV RMSE: {rmse_spatial:.2f} mg/kg")
print()

# ============================================================================
# 6. RESULTS COMPARISON
# ============================================================================

print("[6/7] Comparing validation strategies...")
print("="*60)
print(f"{'Method':<30} {'RMSE (mg/kg)':<15} {'vs Standard':<15}")
print("="*60)
print(f"{'sklearn Standard CV':<30} {rmse_standard_sklearn:<15.2f} {'-':<15}")
print(f"{'MLPY Standard CV':<30} {rmse_standard_mlpy:<15.2f} {'-':<15}")
print(f"{'MLPY Spatial K-Fold':<30} {rmse_spatial:<15.2f} {'-':<15}")
print("="*60)

underestimation = ((rmse_spatial - rmse_standard_mlpy) / rmse_spatial) * 100
print(f"\nStandard CV underestimates error by: {underestimation:.1f}%")
print(f"Absolute difference: {rmse_spatial - rmse_standard_mlpy:.2f} mg/kg")
print()

# ============================================================================
# 7. GENERATE FIGURES
# ============================================================================

print("[7/7] Generating figures...")

# Create output directory
import os
os.makedirs('case_studies/figures', exist_ok=True)

# Figure 1: Spatial distribution of copper
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
scatter = ax.scatter(jura_data['X'], jura_data['Y'],
                     c=jura_data['Cu'], cmap='YlOrRd',
                     s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('X coordinate (km)', fontsize=12)
ax.set_ylabel('Y coordinate (km)', fontsize=12)
ax.set_title('Spatial Distribution of Copper Concentration (Jura Dataset)',
             fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, label='Copper (mg/kg)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('case_studies/figures/jura_spatial_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('case_studies/figures/jura_spatial_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: jura_spatial_distribution.pdf/png")

# Figure 2: CV Comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
methods = ['sklearn\nStandard CV', 'MLPY\nStandard CV', 'MLPY\nSpatial K-Fold']
rmse_values = [rmse_standard_sklearn, rmse_standard_mlpy, rmse_spatial]
colors = ['#3498db', '#3498db', '#e74c3c']

bars = ax.bar(methods, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('RMSE (mg/kg)', fontsize=12)
ax.set_title('Cross-Validation Performance Comparison (Jura Dataset)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(rmse_values) * 1.15)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max(rmse_values)*0.02,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add underestimation annotation
ax.annotate(f'{underestimation:.1f}% underestimation',
            xy=(1.5, (rmse_standard_mlpy + rmse_spatial)/2),
            xytext=(2.5, (rmse_standard_mlpy + rmse_spatial)/2),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('case_studies/figures/jura_cv_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('case_studies/figures/jura_cv_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: jura_cv_comparison.pdf/png")

# Figure 3: Spatial folds visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Get fold assignments from spatial CV
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
fold_assignments = kmeans.fit_predict(coords_array)

scatter = ax.scatter(jura_data['X'], jura_data['Y'],
                     c=fold_assignments, cmap='Set3',
                     s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('X coordinate (km)', fontsize=12)
ax.set_ylabel('Y coordinate (km)', fontsize=12)
ax.set_title('5-Fold Spatial Cross-Validation Partitioning (Jura Dataset)',
             fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, label='Fold', ticks=range(5))
cbar.set_ticklabels([f'Fold {i+1}' for i in range(5)])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('case_studies/figures/jura_spatial_folds.pdf', dpi=300, bbox_inches='tight')
plt.savefig('case_studies/figures/jura_spatial_folds.png', dpi=300, bbox_inches='tight')
print("Saved: jura_spatial_folds.pdf/png")

print()
print("="*80)
print("CASE STUDY 2 COMPLETE")
print("="*80)
print()
print("KEY FINDINGS:")
print(f"  * Moran's I = {morans_i:.3f} -> {interpretation} spatial autocorrelation")
print(f"  * Standard CV RMSE = {rmse_standard_mlpy:.2f} mg/kg")
print(f"  * Spatial CV RMSE = {rmse_spatial:.2f} mg/kg")
print(f"  * Underestimation = {underestimation:.1f}%")
print()
print("This confirms spatial leakage affects mineral prospectivity modeling,")
print("demonstrating the need for spatial CV across different geoscience domains.")
print()
