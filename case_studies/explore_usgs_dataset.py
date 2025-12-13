"""
Explore USGS National Geochemical Survey Dataset
Extract suitable regional subset for mineral prospectivity case study
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

print("="*80)
print("USGS National Geochemical Survey - Dataset Exploration")
print("="*80)
print()

# ============================================================================
# 1. LOAD SHAPEFILE
# ============================================================================

print("[1/6] Loading USGS National Geochemical Survey shapefile...")

# Load the shapefile
gdf = gpd.read_file('case_studies/data/ngs.shp')

print(f"Total samples: {len(gdf)}")
print(f"Columns: {len(gdf.columns)}")
print()

# Display basic info
print("First few column names:")
print(gdf.columns.tolist()[:20])
print()

# Display data types
print("Data types:")
print(gdf.dtypes.head(20))
print()

# ============================================================================
# 2. IDENTIFY COPPER COLUMN AND COORDINATES
# ============================================================================

print("[2/6] Identifying copper concentration and coordinates...")

# Find copper-related columns
copper_cols = [col for col in gdf.columns if 'CU' in col.upper() or 'COPPER' in col.upper()]
print(f"Copper-related columns: {copper_cols}")

# Get coordinates
print(f"Geometry type: {gdf.geometry.geom_type.unique()}")
print(f"CRS: {gdf.crs}")

# Extract coordinates
gdf['lon'] = gdf.geometry.x
gdf['lat'] = gdf.geometry.y

print(f"Longitude range: {gdf['lon'].min():.2f} to {gdf['lon'].max():.2f}")
print(f"Latitude range: {gdf['lat'].min():.2f} to {gdf['lat'].max():.2f}")
print()

# ============================================================================
# 3. EXAMINE COPPER DISTRIBUTION
# ============================================================================

print("[3/6] Examining copper concentration distribution...")

# Assume first copper column is the one we want
if copper_cols:
    cu_col = copper_cols[0]
    print(f"Using column: {cu_col}")

    # Get copper values
    cu_values = gdf[cu_col]

    # Remove missing values
    cu_valid = cu_values[cu_values.notna()]

    print(f"Total samples: {len(cu_values)}")
    print(f"Non-missing values: {len(cu_valid)}")
    print(f"Missing values: {len(cu_values) - len(cu_valid)}")
    print()

    if len(cu_valid) > 0:
        print(f"Copper statistics:")
        print(f"  Min: {cu_valid.min():.2f}")
        print(f"  Max: {cu_valid.max():.2f}")
        print(f"  Mean: {cu_valid.mean():.2f}")
        print(f"  Median: {cu_valid.median():.2f}")
        print(f"  Std: {cu_valid.std():.2f}")
        print()
else:
    print("WARNING: No copper column found!")
    print("Available columns:")
    print(gdf.columns.tolist())
    cu_col = None

# ============================================================================
# 4. FIND SUITABLE REGIONAL SUBSET
# ============================================================================

print("[4/6] Finding suitable regional subset...")

if cu_col:
    # Filter to valid copper values
    gdf_cu = gdf[gdf[cu_col].notna()].copy()

    print(f"Samples with copper data: {len(gdf_cu)}")

    # Strategy: Find region with high copper concentration variability
    # and strong spatial clustering

    # Divide into grid cells and find "hot spots"
    lon_bins = pd.cut(gdf_cu['lon'], bins=20)
    lat_bins = pd.cut(gdf_cu['lat'], bins=20)

    gdf_cu['lon_bin'] = lon_bins
    gdf_cu['lat_bin'] = lat_bins

    # Calculate mean copper per grid cell
    grid_stats = gdf_cu.groupby(['lon_bin', 'lat_bin'])[cu_col].agg(['mean', 'std', 'count'])
    grid_stats = grid_stats[grid_stats['count'] >= 10]  # At least 10 samples

    print(f"Grid cells with 10+ samples: {len(grid_stats)}")

    # Find top 5 regions by copper variability
    top_regions = grid_stats.nlargest(5, 'std')
    print()
    print("Top 5 regions by copper variability:")
    print(top_regions)
    print()

    # Select first region
    if len(top_regions) > 0:
        selected_lon_bin = top_regions.index[0][0]
        selected_lat_bin = top_regions.index[0][1]

        # Extract samples from this region
        subset = gdf_cu[
            (gdf_cu['lon_bin'] == selected_lon_bin) &
            (gdf_cu['lat_bin'] == selected_lat_bin)
        ].copy()

        print(f"Selected region samples: {len(subset)}")
        print(f"Longitude range: {subset['lon'].min():.2f} to {subset['lon'].max():.2f}")
        print(f"Latitude range: {subset['lat'].min():.2f} to {subset['lat'].max():.2f}")
        print(f"Copper range: {subset[cu_col].min():.2f} to {subset[cu_col].max():.2f}")
        print()

        # If subset is too small, expand search
        if len(subset) < 200:
            print("Subset too small, expanding to adjacent cells...")
            # Get center of selected bins
            lon_center = (selected_lon_bin.left + selected_lon_bin.right) / 2
            lat_center = (selected_lat_bin.left + selected_lat_bin.right) / 2

            # Calculate distance from center
            gdf_cu['dist_from_center'] = np.sqrt(
                (gdf_cu['lon'] - lon_center)**2 +
                (gdf_cu['lat'] - lat_center)**2
            )

            # Sort by distance and take closest 300-500 samples
            subset = gdf_cu.nsmallest(400, 'dist_from_center').copy()

            print(f"Expanded subset samples: {len(subset)}")
            print(f"Longitude range: {subset['lon'].min():.2f} to {subset['lon'].max():.2f}")
            print(f"Latitude range: {subset['lat'].min():.2f} to {subset['lat'].max():.2f}")
            print(f"Copper range: {subset[cu_col].min():.2f} to {subset[cu_col].max():.2f}")
            print()

# ============================================================================
# 5. COMPUTE SPATIAL AUTOCORRELATION
# ============================================================================

print("[5/6] Computing spatial autocorrelation for subset...")

if cu_col and len(subset) > 0:
    # Prepare data
    coords = subset[['lon', 'lat']].values
    cu_vals = subset[cu_col].values

    # Compute Moran's I (approximate)
    y = cu_vals
    y_mean = y.mean()
    n = len(y)

    # Calculate pairwise distances
    distances = cdist(coords, coords)

    # Use inverse distance weights (threshold at median distance)
    threshold = np.percentile(distances[distances > 0], 25)  # 25th percentile

    W = np.where((distances < threshold) & (distances > 0), 1.0 / (distances + 0.01), 0)
    np.fill_diagonal(W, 0)
    W_sum = W.sum()

    if W_sum > 0:
        numerator = np.sum(W * np.outer(y - y_mean, y - y_mean))
        denominator = np.sum((y - y_mean)**2)
        morans_i = (n / W_sum) * (numerator / denominator)

        print(f"Moran's I: {morans_i:.3f}")

        interpretation = "strong positive" if morans_i > 0.5 else "moderate positive" if morans_i > 0.3 else "weak positive" if morans_i > 0 else "negative"
        print(f"Interpretation: {interpretation} spatial autocorrelation")
        print()

        if morans_i < 0.3:
            print("WARNING: Spatial autocorrelation is weak. May need different region.")
        else:
            print("✓ Spatial autocorrelation is sufficient for case study!")
    else:
        print("ERROR: Could not compute spatial weights")

# ============================================================================
# 6. SAVE SUBSET FOR CASE STUDY
# ============================================================================

print("[6/6] Saving subset for case study...")

if cu_col and len(subset) > 0:
    # Prepare final dataset
    # Rename copper column to 'Cu' for consistency
    final_data = subset[['lon', 'lat', cu_col]].copy()
    final_data.columns = ['X', 'Y', 'Cu']

    # Add index
    final_data['sample_id'] = range(len(final_data))

    # Save to CSV
    output_file = 'case_studies/data/usgs_copper_subset.csv'
    final_data.to_csv(output_file, index=False)

    print(f"✓ Saved {len(final_data)} samples to: {output_file}")
    print()
    print("Dataset summary:")
    print(final_data.describe())
    print()

    # Create quick visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Spatial distribution
    scatter = axes[0].scatter(final_data['X'], final_data['Y'],
                              c=final_data['Cu'], cmap='YlOrRd',
                              s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Spatial Distribution of Copper Concentration')
    plt.colorbar(scatter, ax=axes[0], label='Copper (ppm)')
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(final_data['Cu'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Copper (ppm)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Copper Concentration')
    axes[1].axvline(final_data['Cu'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {final_data["Cu"].mean():.1f}')
    axes[1].axvline(final_data['Cu'].median(), color='blue', linestyle='--',
                    linewidth=2, label=f'Median: {final_data["Cu"].median():.1f}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('case_studies/figures/usgs_copper_exploration.png', dpi=150, bbox_inches='tight')
    print("✓ Saved exploration figure: case_studies/figures/usgs_copper_exploration.png")

print()
print("="*80)
print("DATASET EXPLORATION COMPLETE")
print("="*80)
