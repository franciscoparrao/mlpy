"""
Extract mining region with strong spatial autocorrelation from USGS data
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

print("="*80)
print("USGS Geochemical Survey - Mining Region Extraction")
print("="*80)
print()

# Load shapefile
print("[1/5] Loading shapefile...")
gdf = gpd.read_file('case_studies/data/ngs.shp')
print(f"Total samples: {len(gdf)}")

# Get copper column
cu_col = 'CU_ICP40'
gdf['lon'] = gdf.geometry.x
gdf['lat'] = gdf.geometry.y

# Filter valid copper values
gdf_cu = gdf[gdf[cu_col].notna()].copy()
print(f"Samples with copper: {len(gdf_cu)}")
print()

# Test specific mining regions
print("[2/5] Testing known mining regions...")

regions = [
    {"name": "Arizona Copper Belt", "lon": (-110, -108), "lat": (31, 34)},
    {"name": "Montana/Idaho", "lon": (-114, -111), "lat": (45, 48)},
    {"name": "Nevada Mining", "lon": (-118, -115), "lat": (38, 41)},
    {"name": "Alaska Interior", "lon": (-150, -145), "lat": (60, 65)},
    {"name": "Colorado Mineral Belt", "lon": (-107, -105), "lat": (38, 40)},
]

best_subset = None
best_morans_i = -999
best_region_name = None

for region in regions:
    print(f"\nTesting: {region['name']}")

    # Extract region
    subset = gdf_cu[
        (gdf_cu['lon'] >= region['lon'][0]) &
        (gdf_cu['lon'] <= region['lon'][1]) &
        (gdf_cu['lat'] >= region['lat'][0]) &
        (gdf_cu['lat'] <= region['lat'][1])
    ].copy()

    n_samples = len(subset)
    print(f"  Samples: {n_samples}")

    if n_samples < 100:
        print("  Too few samples")
        continue

    # Sample if too large
    if n_samples > 500:
        subset_test = subset.sample(n=500, random_state=42)
    else:
        subset_test = subset

    # Calculate Moran's I
    coords = subset_test[['lon', 'lat']].values
    cu_vals = subset_test[cu_col].values

    y = cu_vals
    y_mean = y.mean()
    n = len(y)

    distances = cdist(coords, coords)
    threshold = np.percentile(distances[distances > 0], 20)

    W = np.where((distances < threshold) & (distances > 0), 1.0 / (distances + 0.001), 0)
    np.fill_diagonal(W, 0)
    W_sum = W.sum()

    if W_sum > 0:
        numerator = np.sum(W * np.outer(y - y_mean, y - y_mean))
        denominator = np.sum((y - y_mean)**2)
        morans_i = (n / W_sum) * (numerator / denominator)

        print(f"  Copper: {subset[cu_col].min():.1f} to {subset[cu_col].max():.1f} ppm")
        print(f"  Moran's I: {morans_i:.3f}")

        if morans_i > best_morans_i:
            best_morans_i = morans_i
            best_subset = subset.copy()
            best_region_name = region['name']

print()
print("="*80)

if best_subset is not None:
    subset = best_subset
    print(f"[3/5] Selected: {best_region_name}")
    print(f"  Moran's I: {best_morans_i:.3f}")
    print(f"  Samples: {len(subset)}")
    print(f"  Copper range: {subset[cu_col].min():.1f} - {subset[cu_col].max():.1f} ppm")
    print()

    # Reduce to manageable size if needed
    if len(subset) > 600:
        print(f"  Reducing to 500 samples...")
        subset = subset.sample(n=500, random_state=42)

    # Save dataset
    print("[4/5] Saving dataset...")
    final_data = subset[['lon', 'lat', cu_col]].copy()
    final_data.columns = ['X', 'Y', 'Cu']
    final_data = final_data.reset_index(drop=True)

    output_file = 'case_studies/data/usgs_copper_mining.csv'
    final_data.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    print()
    print(final_data.describe())
    print()

    # Create visualization
    print("[5/5] Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Spatial distribution
    scatter = axes[0].scatter(final_data['X'], final_data['Y'],
                              c=final_data['Cu'], cmap='YlOrRd',
                              s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title(f'Copper Distribution - {best_region_name}')
    plt.colorbar(scatter, ax=axes[0], label='Copper (ppm)')
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(final_data['Cu'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Copper (ppm)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Copper Concentration Distribution')
    axes[1].axvline(final_data['Cu'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {final_data["Cu"].mean():.1f}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_file = 'case_studies/figures/usgs_mining_region.png'
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_file}")

    print()
    print("="*80)
    print("EXTRACTION COMPLETE")
    print(f"Region: {best_region_name}")
    print(f"Spatial autocorrelation (Moran's I): {best_morans_i:.3f}")
    print(f"Status: {'GOOD' if best_morans_i > 0.3 else 'WEAK'}")
    print("="*80)

else:
    print("ERROR: No suitable region found")
