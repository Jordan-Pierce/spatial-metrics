# Spatial Metrics

A comprehensive Python library for computing spatial metrics from geospatial data, designed for deep-sea mining operations and environmental impact assessments.

## Overview

`SpatialMetricsAnalyzer` computes a wide range of spatial metrics from:
- **Orthorectified imagery** (optical/acoustic)
- **Elevation/depth rasters**
- **Binary segmentation masks**
- **GeoJSON polygon annotations**

The analyzer automatically detects coordinate reference systems (CRS) from rasters to determine scale, or uses a user-provided `scale_factor` for geographic/missing CRS.

## Installation

```bash
pip install numpy pandas rasterio scipy shapely tqdm
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.analyzer import SpatialMetricsAnalyzer, analyze

# Full analysis with report generation
analyzer = SpatialMetricsAnalyzer(
    mask_path="nodules.tif",
    geojson_path="annotations.geojson",
    elevation_path="depth.tif",  # Optional, for 3D metrics
    scale_factor=0.01  # 1cm per pixel (if CRS not projected)
)

# Generate comprehensive report
summary_df, report = analyzer.generate_report(output_path="results.json")

# Or use convenience function
summary_df, report = analyze(
    mask_path="nodules.tif",
    geojson_path="annotations.geojson",
    output_path="results.json"
)
```

## Metrics Reference

### Density & Abundance (The "How Much")

These metrics answer: *"What is the quantity of material present?"*

#### `calculate_pcf()` — Pixel Coverage Fraction

**The Visual Density Metric**

Calculates the percentage of the image/ROI covered by target objects.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `mask_path` |
| **Formula** | `PCF = (Sum of mask pixels) / (Total pixels) × 100` |

**Returns:**
| Key | Description |
|-----|-------------|
| `pcf_percent` | Coverage as percentage (0-100) |
| `pcf_fraction` | Coverage as fraction (0-1) |
| `covered_area_m2` | Covered area in square meters |
| `total_area_m2` | Total area in square meters |

**Example:**
```python
pcf = analyzer.calculate_pcf()
print(f"Coverage: {pcf['pcf_percent']:.2f}%")
# Output: Coverage: 12.34%
```

---

#### `calculate_abundance()` — Resource Density

**The Economic Grade Metric**

Counts the number of distinct objects per unit area.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `geojson_path` |
| **Formula** | `Abundance = (Number of polygons) / (Total area in m²)` |

**Returns:**
| Key | Description |
|-----|-------------|
| `count` | Total number of objects |
| `abundance_per_m2` | Objects per square meter |
| `abundance_per_100m2` | Objects per 100 square meters |

**Example:**
```python
abundance = analyzer.calculate_abundance()
print(f"Density: {abundance['abundance_per_m2']:.2f} objects/m²")
```

---

#### `calculate_spatial_homogeneity(grid_size=4)` — Patchiness

**The Variance-to-Mean Ratio (VMR) Metric**

Quantifies how uniformly objects are distributed using quadrat analysis.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `geojson_path` |
| `grid_size` | Grid cells per dimension (default: 4) |
| **Formula** | `VMR = Variance(counts) / Mean(counts)` |

**Interpretation:**
| VMR Value | Pattern |
|-----------|---------|
| VMR ≈ 1 | Random (Poisson) distribution |
| VMR < 1 | Uniform/regular distribution |
| VMR > 1 | Clustered/aggregated distribution |

**Returns:**
| Key | Description |
|-----|-------------|
| `vmr` | Variance-to-Mean Ratio |
| `pattern` | Interpreted pattern ('clustered', 'random', 'uniform') |
| `mean_count` | Mean objects per cell |
| `cell_counts` | List of counts per cell |

**Example:**
```python
homogeneity = analyzer.calculate_spatial_homogeneity(grid_size=5)
print(f"Pattern: {homogeneity['pattern']} (VMR={homogeneity['vmr']:.2f})")
# Output: Pattern: CLUSTERED (VMR=2.45)
```

---

### Proximity & Clustering (The "How Close")

These metrics describe *"how objects are arranged"* — critical for machine interactions and path planning.

#### `calculate_nearest_neighbor_distance(method='edge')` — Physical Gap

**The Nearest Neighbor Distance (NND) Metric**

Measures the distance from each object to its closest neighbor.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `geojson_path` |
| `method` | `'edge'` (edge-to-edge) or `'centroid'` (center-to-center) |

**Methods:**
| Method | Description | Use Case |
|--------|-------------|----------|
| `'edge'` | Shortest distance between polygon boundaries | Engineering (jamming prediction) |
| `'centroid'` | Distance between polygon centers | General spatial analysis (faster) |

**Returns:**
| Key | Description |
|-----|-------------|
| `mean_nnd_m` | Mean nearest neighbor distance (meters) |
| `median_nnd_m` | Median distance (meters) |
| `min_nnd_m` / `max_nnd_m` | Range of distances |
| `nnd_values_m` | All individual NND values |

**Example:**
```python
# Edge-to-edge for engineering analysis
nnd_edge = analyzer.calculate_nearest_neighbor_distance(method='edge')
print(f"Mean gap: {nnd_edge['mean_nnd_m']:.3f} m")

# Centroid for faster analysis
nnd_centroid = analyzer.calculate_nearest_neighbor_distance(method='centroid')
```

> **Note:** For jamming prediction, use `method='edge'`. Two objects might have centroids 15cm apart, but edges only 3cm apart — a critical difference for collector design.

---

#### `calculate_passability_index()` — Navigable Space

**The Corridor Width Metric**

Analyzes void space to determine the maximum vehicle size that can navigate without collision.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `mask_path` |
| **Method** | Euclidean Distance Transform (EDT) on inverted mask |

**Returns:**
| Key | Description |
|-----|-------------|
| `max_passage_diameter_m` | Largest clear passage (meters) |
| `mean_clearance_m` | Average distance to nearest obstacle |
| `passability_fraction` | Fraction of area that is passable |

**Example:**
```python
passability = analyzer.calculate_passability_index()
print(f"Max vehicle width: {passability['max_passage_diameter_m']:.2f} m")
print(f"Free space: {passability['passability_fraction']*100:.1f}%")
```

---

#### `calculate_ripleys_k(radii=None, n_radii=20)` — Scale of Aggregation

**The Multi-Scale Clustering Metric**

Reveals at what spatial scale(s) objects are clustered.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `geojson_path` |
| `radii` | List of radii in meters (auto-generated if None) |
| `n_radii` | Number of radii to evaluate (default: 20) |

**Formula:**
```
K(r) = (Area/n²) × Σᵢ Σⱼ I(dᵢⱼ ≤ r)
L(r) = sqrt(K(r)/π) - r  (normalized)
```

**Interpretation:**
| L(r) Value | Meaning |
|------------|---------|
| L(r) > 0 | Clustering at scale r |
| L(r) ≈ 0 | Random at scale r |
| L(r) < 0 | Dispersion at scale r |

**Returns:**
| Key | Description |
|-----|-------------|
| `radii_m` | Evaluated radii |
| `k_observed` / `k_expected` | K function values |
| `l_function` | Normalized L values |
| `max_clustering_radius_m` | Radius of peak clustering |

**Example:**
```python
ripley = analyzer.calculate_ripleys_k(n_radii=30)
print(f"Peak clustering at: {ripley['max_clustering_radius_m']:.2f} m")
```

> A peak at r=2m indicates nodules arranged in patches roughly 2m wide.

---

#### `calculate_clark_evans()` — Clustering Fingerprint

**The Single-Value Aggregation Index**

Summarizes overall clustering intensity in one number.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `geojson_path` |
| **Formula** | `R = r̄_observed / r̄_expected` |

**Interpretation:**
| R Value | Pattern |
|---------|---------|
| R = 1.0 | Random distribution |
| R > 1.0 | Dispersed (further apart than random) |
| R < 1.0 | Clustered (closer than random) |
| R → 0 | Maximum clustering |

**Returns:**
| Key | Description |
|-----|-------------|
| `clark_evans_r` | The R index (0 to ~2) |
| `pattern` | 'clustered', 'random', or 'dispersed' |
| `z_score` | Statistical significance |

**Example:**
```python
ce = analyzer.calculate_clark_evans()
print(f"Clark-Evans R: {ce['clark_evans_r']:.3f} ({ce['pattern']})")
# Output: Clark-Evans R: 0.724 (clustered)
```

---

### Individual Morphology (The "What Kind")

Shape descriptors characterizing individual objects.

#### `calculate_morphology_stats()` — Shape Characterization

**Per-Object Morphology Analysis**

Computes three key shape metrics for each polygon:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Circularity** | `4π × Area / Perimeter²` | 1.0 = circle, <0.5 = irregular |
| **Solidity** | `Area / Convex_Hull_Area` | 1.0 = convex, low = jagged |
| **OBB Aspect Ratio** | `min(width,height) / max(width,height)` | 1.0 = square, →0 = elongated |

| Parameter | Description |
|-----------|-------------|
| **Requires** | `geojson_path` |

**Returns:** `pandas.DataFrame` with columns:
| Column | Description |
|--------|-------------|
| `polygon_id` | Unique identifier |
| `area_m2` | Area in square meters |
| `perimeter_m` | Perimeter in meters |
| `circularity` | Roundness (0-1) |
| `solidity` | Smoothness (0-1) |
| `obb_aspect_ratio` | Elongation (0-1) |
| `centroid_x_m`, `centroid_y_m` | Location |

**Example:**
```python
morphology = analyzer.calculate_morphology_stats()
print(f"Mean circularity: {morphology['circularity'].mean():.3f}")

# Filter for round nodules
round_nodules = morphology[morphology['circularity'] > 0.8]
print(f"Found {len(round_nodules)} round objects")
```

---

### Verticality & Interaction (The "3rd Dimension")

These metrics require elevation/depth data.

#### `calculate_protrusion()` — Stick-up Height

**The Collector Clearance Metric**

Measures how far objects protrude above the local seafloor.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `elevation_path`, `mask_path` |
| **Method** | Estimate seafloor plane → measure height above |

**Returns:**
| Key | Description |
|-----|-------------|
| `mean_protrusion_m` | Mean stick-up height (meters) |
| `max_protrusion_m` | Maximum height |
| `seafloor_elevation_m` | Estimated baseline |
| `per_object_stats` | Per-polygon heights |

**Example:**
```python
protrusion = analyzer.calculate_protrusion()
print(f"Mean stick-up: {protrusion['mean_protrusion_m']*100:.1f} cm")
```

> If mean stick-up is 2cm, collector must be lowered aggressively (risk sediment intake).
> If stick-up is 10cm, collector can "clip" nodules off the top.

---

#### `calculate_3d_rugosity()` — Surface Texture

**The Surface Complexity Metric**

Compares true 3D surface area to projected 2D footprint.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `elevation_path`, `mask_path` |
| **Formula** | `Rugosity = 3D_Surface_Area / 2D_Area` |

**Interpretation:**
| Value | Surface Type |
|-------|--------------|
| 1.0 | Perfectly flat |
| 1.0-1.2 | Smooth (river stone) |
| 1.2-1.5 | Moderate texture |
| >1.5 | Highly complex (coral-like) |

**Returns:**
| Key | Description |
|-----|-------------|
| `rugosity_index` | 3D/2D area ratio (≥1.0) |
| `mean_slope_degrees` | Average surface slope |
| `surface_area_3d_m2` | True surface area |

**Example:**
```python
rugosity = analyzer.calculate_3d_rugosity()
print(f"Surface complexity: {rugosity['rugosity_index']:.2f}x flat")
```

---

### Ecosystem Dynamics (The "Effect on Critters")

Metrics for environmental impact assessment.

#### `calculate_biodiversity_correlation(...)` — Habitat Association

**The Biodiversity-Density Correlation Metric**

Tests whether biological abundance correlates with resource density.

| Parameter | Description |
|-----------|-------------|
| **Requires** | `geojson_path` with multi-class annotations |
| `resource_class` | Class name for resources (default: 'nodule') |
| `biology_class` | Class name for organisms (default: 'organism') |
| `grid_size` | Grid cells per dimension (default: 5) |
| `method` | 'pearson' or 'spearman' (default: 'pearson') |

**Interpretation:**
| Correlation | Meaning |
|-------------|---------|
| ~1.0 | Strong positive: "More Rocks == More Life" (conflict) |
| ~0 | No association: biology exists independently |
| ~-1.0 | Strong negative: biology avoids resource areas |

**Returns:**
| Key | Description |
|-----|-------------|
| `correlation` | Correlation coefficient (-1 to 1) |
| `p_value` | Statistical significance |
| `interpretation` | Text explanation |

**Example:**
```python
bio_corr = analyzer.calculate_biodiversity_correlation(
    resource_class='nodule',
    biology_class='sponge',
    method='spearman'
)
print(f"Correlation: {bio_corr['correlation']:.3f}")
print(f"Interpretation: {bio_corr['interpretation']}")
```

---

### Report Generation

#### `generate_report(output_path=None, include_morphology_details=True)`

**Comprehensive Analysis Report**

Automatically runs all feasible metrics based on available inputs.

| Parameter | Description |
|-----------|-------------|
| `output_path` | Path to save JSON report (optional) |
| `include_morphology_details` | Include per-polygon data (default: True) |

**Returns:** `Tuple[pd.DataFrame, Dict]`
- `DataFrame`: Summary table with key metrics
- `Dict`: Full results (JSON-serializable)

**Output Files:**
- `{output_path}.json` — Full report
- `{output_path}.csv` — Summary table

**Example:**
```python
summary_df, report = analyzer.generate_report(output_path="analysis.json")

# Access specific metric
pcf = report['metrics']['pcf']['pcf_percent']

# Check what was computed
print(f"Computed: {report['computed_metrics']}")
print(f"Skipped: {report['skipped_metrics']}")
```

---

## Data Requirements

| Metric Category | Mask | GeoJSON | Elevation |
|-----------------|:----:|:-------:|:---------:|
| PCF | ✓ | | |
| Abundance | | ✓ | |
| Spatial Homogeneity | | ✓ | |
| Nearest Neighbor | | ✓ | |
| Passability | ✓ | | |
| Ripley's K | | ✓ | |
| Clark-Evans | | ✓ | |
| Morphology | | ✓ | |
| Protrusion | ✓ | | ✓ |
| 3D Rugosity | ✓ | | ✓ |
| Biodiversity Corr. | | ✓* | |

\* Requires multi-class GeoJSON with both resource and biology labels

---

## Scale Detection

The analyzer automatically detects scale from raster CRS:

1. **Projected CRS** (e.g., UTM): `meters_per_pixel` calculated from affine transform
2. **Geographic CRS** (e.g., WGS84): Uses provided `scale_factor`
3. **No CRS**: Uses provided `scale_factor` (default: 1.0)

```python
# For projected data (auto-detected)
analyzer = SpatialMetricsAnalyzer(mask_path="utm_raster.tif")

# For unprojected data (manual scale)
analyzer = SpatialMetricsAnalyzer(
    mask_path="image.tif",
    scale_factor=0.01  # 1 cm per pixel
)
```

---

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Citation

If you use this library in research, please cite:

```bibtex
@software{spatial_metrics,
  title = {Spatial Metrics Analyzer},
  author = {Pierce, Jordan},
  year = {2026},
  url = {https://github.com/Jordan-Pierce/spatial-metrics}
}
```