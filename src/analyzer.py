"""
SpatialMetricsAnalyzer: A comprehensive spatial metrics analysis tool for deep-sea mining.

This module provides the SpatialMetricsAnalyzer class for computing spatial metrics
from orthorectified optical/acoustic imagery, elevation rasters, and semantic segmentation masks.

SEMANTIC SEGMENTATION MASK STRUCTURE:
    The analyzer expects semantic segmentation masks with the following class definitions:
    - 0: Background (non-labeled, outside ROI)
    - 1: Substrate/Sediment (navigable surface, reference for metrics)
    - 2: Nodule (target mining object, obstacle for collision avoidance)
    - 3: Organisms (living creatures, obstacle to preserve and avoid)
    
    Substrate (mask==1) is treated as free/navigable space for passability calculations.
    Objects (mask>=2) are treated as obstacles in collision avoidance metrics.

The metrics are organized into five categories:
    - Density & Abundance: PCF, Resource Density, Spatial Homogeneity
    - Proximity & Clustering: NND, Passability, Ripley's K, Clark-Evans
    - Individual Morphology: Circularity, Solidity, OBB Aspect Ratio
    - Verticality & Interaction: Protrusion, 3D Rugosity
    - Ecosystem Dynamics: Biodiversity-Density Correlation

Author: Jordan Pierce
Date: January 2026
"""

import json
import warnings
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from scipy import ndimage
from scipy.spatial import KDTree
from scipy.stats import pearsonr, spearmanr

from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

import rasterio
from rasterio.crs import CRS

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# CLASS DEFINITION
# =============================================================================


class SpatialMetricsAnalyzer:
    """
    A comprehensive analyzer for computing spatial metrics from geospatial data.
    
    This class handles optical/acoustic imagery, elevation rasters, binary masks,
    and GeoJSON annotations to compute a wide range of spatial metrics relevant
    to deep-sea mining operations and environmental impact assessments.
    
    The analyzer automatically detects the coordinate reference system (CRS) from
    raster inputs to determine the appropriate scale factor (meters per pixel).
    If the CRS is geographic or missing, a user-provided scale_factor is used.
    
    Attributes:
        image_path (Path): Path to the optical/acoustic image raster.
        elevation_path (Path): Path to the elevation/depth raster.
        mask_path (Path): Path to the binary segmentation mask raster.
        geojson_path (Path): Path to the GeoJSON annotations file.
        scale_factor (float): Meters per pixel conversion factor.
        meters_per_pixel (float): Calculated or provided scale in meters.
        
    Example:
        >>> analyzer = SpatialMetricsAnalyzer(
        ...     mask_path="mask.tif",
        ...     geojson_path="annotations.geojson",
        ...     scale_factor=0.01  # 1cm per pixel
        ... )
        >>> report = analyzer.generate_report(output_path="results.json")
    """
    
    def __init__(
        self,
        image_path: Optional[str] = None,
        elevation_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        geojson_path: Optional[str] = None,
        scale_factor: float = 1.0
    ):
        """
        Initialize the SpatialMetricsAnalyzer with optional data inputs.
        
        Args:
            image_path: Path to the optical/acoustic image raster (optional).
                       Used for visualization and context.
            elevation_path: Path to the elevation/depth raster (optional).
                           Required for 3D metrics (protrusion, 3D rugosity).
            mask_path: Path to the binary segmentation mask raster (optional).
                      Required for pixel-based metrics (PCF, passability).
            geojson_path: Path to the GeoJSON file with polygon annotations (optional).
                         Required for morphology and clustering metrics.
            scale_factor: Default meters-per-pixel if CRS is geographic or missing.
                         Defaults to 1.0 (assumes pixel units = meters).
        
        Raises:
            FileNotFoundError: If any provided file path does not exist.
            ValueError: If scale_factor is not positive.
        
        Note:
            The analyzer will detect projected CRS (e.g., UTM) from rasters and
            calculate meters_per_pixel automatically from the affine transform.
            For geographic CRS or missing CRS, the scale_factor parameter is used.
        """
        # Validate scale factor
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        
        # Store paths (convert to Path objects if provided)
        self.image_path = Path(image_path) if image_path else None
        self.elevation_path = Path(elevation_path) if elevation_path else None
        self.mask_path = Path(mask_path) if mask_path else None
        self.geojson_path = Path(geojson_path) if geojson_path else None
        
        # Validate file existence
        for path, name in [
            (self.image_path, "image"),
            (self.elevation_path, "elevation"),
            (self.mask_path, "mask"),
            (self.geojson_path, "geojson")
        ]:
            if path is not None and not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")
        
        # Initialize scale factor
        self.scale_factor = scale_factor
        self.meters_per_pixel = self._detect_scale()
        
        # Cache for loaded data
        self._mask_data: Optional[np.ndarray] = None
        self._elevation_data: Optional[np.ndarray] = None
        self._elevation_transform: Optional[rasterio.Affine] = None
        self._polygons: Optional[List[Polygon]] = None
        self._raster_shape: Optional[Tuple[int, int]] = None
        self._raster_transform: Optional[rasterio.Affine] = None
        
        # Print initialization summary
        print("=" * 60)
        print("SpatialMetricsAnalyzer Initialized")
        print("=" * 60)
        print(f"  Image:      {self.image_path or 'Not provided'}")
        print(f"  Elevation:  {self.elevation_path or 'Not provided'}")
        print(f"  Mask:       {self.mask_path or 'Not provided'}")
        print(f"  GeoJSON:    {self.geojson_path or 'Not provided'}")
        print(f"  Scale:      {self.meters_per_pixel:.6f} meters/pixel")
        print("=" * 60)
    
    def _detect_scale(self) -> float:
        """
        Detect the scale (meters per pixel) from raster CRS.
        
        This method checks available rasters for their coordinate reference system.
        If a projected CRS (e.g., UTM) is found, the pixel size is extracted from
        the affine transform. Otherwise, the user-provided scale_factor is used.
        
        Returns:
            float: The meters per pixel scale factor.
        
        Note:
            Priority order for CRS detection: mask -> elevation -> image
        """
        # Try to detect CRS from available rasters (priority: mask > elevation > image)
        raster_paths = [self.mask_path, self.elevation_path, self.image_path]
        
        for raster_path in raster_paths:
            if raster_path is None:
                continue
                
            try:
                with rasterio.open(raster_path) as src:
                    crs = src.crs
                    transform = src.transform
                    
                    # Check if CRS is projected (has linear units like meters)
                    if crs is not None and crs.is_projected:
                        # Extract pixel size from affine transform
                        # transform.a is the pixel width, transform.e is pixel height (negative)
                        pixel_size_x = abs(transform.a)
                        pixel_size_y = abs(transform.e)
                        
                        # Use average if not square pixels
                        meters_per_pixel = (pixel_size_x + pixel_size_y) / 2
                        
                        print(f"[INFO] Detected projected CRS: {crs}")
                        print(f"[INFO] Calculated scale: {meters_per_pixel:.6f} m/px")
                        
                        return meters_per_pixel
                    else:
                        print(f"[INFO] CRS is geographic or missing, using scale_factor")
                        
            except Exception as e:
                print(f"[WARNING] Could not read CRS from {raster_path}: {e}")
        
        # Fallback to user-provided scale factor
        print(f"[INFO] Using provided scale_factor: {self.scale_factor} m/px")
        return self.scale_factor
    
    def _load_mask(self) -> np.ndarray:
        """
        Load and cache the semantic segmentation mask raster.
        
        Mask class definitions:
            - 0: Background (non-labeled, outside ROI)
            - 1: Substrate/Sediment (navigable free space)
            - 2: Nodule (obstacle, target object for mining)
            - 3: Organisms (obstacle, living creatures to avoid)
        
        Returns:
            np.ndarray: Semantic mask array with values in [0, 1, 2, 3].
        
        Raises:
            ValueError: If mask_path was not provided.
        """
        if self._mask_data is not None:
            return self._mask_data
        
        if self.mask_path is None:
            raise ValueError("Mask path not provided. Cannot compute mask-based metrics.")
        
        print("[INFO] Loading semantic segmentation mask raster...")
        with rasterio.open(self.mask_path) as src:
            self._mask_data = src.read(1).astype(np.uint8)  # Preserve semantic classes
            self._raster_shape = self._mask_data.shape
            self._raster_transform = src.transform
            
            # Validate mask contains expected semantic classes (0, 1, 2, 3)
            unique_classes = np.unique(self._mask_data)
            print(f"[INFO] Semantic classes present: {unique_classes}")
        
        print(f"[INFO] Semantic mask loaded: shape={self._raster_shape}")
        return self._mask_data
    
    def _load_elevation(self) -> np.ndarray:
        """
        Load and cache the elevation/depth raster.
        
        Returns:
            np.ndarray: Elevation values array.
        
        Raises:
            ValueError: If elevation_path was not provided.
        
        Note:
            Elevation and mask may have different pixel dimensions as long as they
            cover the same geographic area. Use _resample_elevation_to_mask() when
            pixel-wise alignment is required.
        """
        if self._elevation_data is not None:
            return self._elevation_data
        
        if self.elevation_path is None:
            raise ValueError("Elevation path not provided. Cannot compute 3D metrics.")
        
        print("[INFO] Loading elevation raster...")
        with rasterio.open(self.elevation_path) as src:
            self._elevation_data = src.read(1).astype(np.float32)
            self._elevation_transform = src.transform
            
            # Handle nodata values
            nodata = src.nodata
            if nodata is not None:
                self._elevation_data[self._elevation_data == nodata] = np.nan

        print(f"[INFO] Elevation loaded: shape={self._elevation_data.shape}")
        return self._elevation_data
    
    def _resample_elevation_to_mask(self, elevation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Resample elevation raster to match mask dimensions using bilinear interpolation.
        
        This allows elevation and mask to cover the same geographic area at different
        resolutions. The elevation will be resampled to match the mask's pixel grid.
        
        Args:
            elevation: Source elevation array (H1 x W1)
            mask: Target mask array (H2 x W2)
        
        Returns:
            np.ndarray: Resampled elevation array with shape matching mask (H2 x W2)
        
        Note:
            Uses scipy.ndimage.zoom with bilinear interpolation. NaN values are
            preserved during resampling.
        """
        # Delegate to the more general shape-resampling helper
        return self._resample_elevation_to_shape(elevation, mask.shape)

    def _resample_elevation_to_shape(self, elevation: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resample elevation array to a given target shape (H, W) using bilinear interpolation.

        Preserves NaN values and prints informative messages about zoom factors.
        """
        if elevation.shape == target_shape:
            return elevation

        from scipy.ndimage import zoom

        zoom_factors = (
            target_shape[0] / elevation.shape[0],
            target_shape[1] / elevation.shape[1]
        )

        print(f"[INFO] Resampling elevation from {elevation.shape} to {target_shape} "
              f"(zoom factors: {zoom_factors[0]:.3f}, {zoom_factors[1]:.3f})")

        # Use bilinear interpolation (order=1) and preserve NaNs
        # scipy.ndimage.zoom does not handle NaNs specially; preserve mask
        nan_mask = np.isnan(elevation)

        # Temporarily fill NaNs with zero to avoid propagation during zoom
        filled = np.where(nan_mask, 0.0, elevation)

        resampled_filled = zoom(filled, zoom_factors, order=1, mode='nearest')

        # Resample the nan mask using nearest-neighbor to identify NaN regions
        resampled_mask = zoom(~nan_mask.astype(np.uint8), zoom_factors, order=0, mode='nearest')

        # Where resampled_mask == 0, set NaN
        resampled = resampled_filled.astype(np.float32)
        resampled[resampled_mask == 0] = np.nan

        return resampled
    
    def _load_polygons(self) -> List[Polygon]:
        """
        Load and cache polygons from GeoJSON file.
        
        Returns:
            List[Polygon]: List of Shapely polygon geometries.
        
        Raises:
            ValueError: If geojson_path was not provided or file is invalid.
        """
        if self._polygons is not None:
            return self._polygons
        
        if self.geojson_path is None:
            raise ValueError("GeoJSON path not provided. Cannot compute polygon-based metrics.")
        
        print("[INFO] Loading GeoJSON polygons...")
        with open(self.geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        self._polygons = []
        features = geojson_data.get('features', [])
        
        for feature in tqdm(features, desc="Parsing polygons", unit="feature"):
            geom = feature.get('geometry')
            if geom is None:
                continue
            
            try:
                poly = shape(geom)
                # Handle both Polygon and MultiPolygon
                if isinstance(poly, Polygon) and poly.is_valid:
                    self._polygons.append(poly)
                elif isinstance(poly, MultiPolygon):
                    for p in poly.geoms:
                        if p.is_valid:
                            self._polygons.append(p)
            except Exception as e:
                print(f"[WARNING] Skipping invalid geometry: {e}")
        
        print(f"[INFO] Loaded {len(self._polygons)} valid polygons")
        return self._polygons
    
    def _load_polygons_with_classes(self) -> Dict[str, List[Polygon]]:
        """
        Load polygons from GeoJSON grouped by class/category.
        
        Returns:
            Dict[str, List[Polygon]]: Dictionary mapping class names to polygon lists.
        
        Raises:
            ValueError: If geojson_path was not provided.
        """
        if self.geojson_path is None:
            raise ValueError("GeoJSON path not provided.")
        
        print("[INFO] Loading GeoJSON polygons with class labels...")
        with open(self.geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        polygons_by_class: Dict[str, List[Polygon]] = {}
        features = geojson_data.get('features', [])
        
        for feature in tqdm(features, desc="Parsing polygons", unit="feature"):
            geom = feature.get('geometry')
            props = feature.get('properties', {})
            
            if geom is None:
                continue
            
            # Try common class field names
            class_name = (
                props.get('class') or 
                props.get('category') or 
                props.get('label') or 
                props.get('type') or 
                props.get('short_label_code') or
                props.get('long_label_code') or
                'unknown'
            )
            
            try:
                poly = shape(geom)
                if isinstance(poly, Polygon) and poly.is_valid:
                    if class_name not in polygons_by_class:
                        polygons_by_class[class_name] = []
                    polygons_by_class[class_name].append(poly)
                elif isinstance(poly, MultiPolygon):
                    for p in poly.geoms:
                        if p.is_valid:
                            if class_name not in polygons_by_class:
                                polygons_by_class[class_name] = []
                            polygons_by_class[class_name].append(p)
            except Exception as e:
                print(f"[WARNING] Skipping invalid geometry: {e}")
        
        print(f"[INFO] Found classes: {list(polygons_by_class.keys())}")
        return polygons_by_class
    
    # =========================================================================
    # DENSITY & ABUNDANCE METRICS
    # =========================================================================
    
    def calculate_pcf(self) -> Dict[str, float]:
        """
        Calculate Pixel Coverage Fraction (PCF) - The Visual Density Metric.
        
        PCF represents the percentage of the image/ROI covered by target objects.
        It is the most basic density metric, calculable from a 2D segmentation
        mask without any scale calibration.
        
        Formula:
            PCF = (Sum of mask pixels) / (Total ROI pixels) × 100
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'pcf_percent': Coverage as percentage (0-100)
                - 'pcf_fraction': Coverage as fraction (0-1)
                - 'covered_pixels': Number of pixels covered by objects
                - 'total_pixels': Total number of pixels in ROI
                - 'covered_area_m2': Covered area in square meters
                - 'total_area_m2': Total area in square meters
        
        Raises:
            ValueError: If mask_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(mask_path="mask.tif")
            >>> pcf = analyzer.calculate_pcf()
            >>> print(f"Coverage: {pcf['pcf_percent']:.2f}%")
        
        Note:
            This metric represents "visual clutter" - the proportion of the
            field of view occupied by the target class (e.g., nodules).
        """
        print("\n[METRIC] Calculating Pixel Coverage Fraction (PCF)...")
        
        # Load mask data
        mask = self._load_mask()
        
        # Calculate coverage (count nodules and organisms, excluding background and substrate)
        # Mask values: 0=background, 1=substrate, 2=nodule, 3=organisms
        # Target objects are nodules (2) and organisms (3)
        covered_pixels = np.sum(mask >= 2)
        total_pixels = mask.size
        
        # Calculate fraction and percentage
        pcf_fraction = covered_pixels / total_pixels
        pcf_percent = pcf_fraction * 100
        
        # Convert to physical area (square meters)
        pixel_area_m2 = self.meters_per_pixel ** 2
        covered_area_m2 = covered_pixels * pixel_area_m2
        total_area_m2 = total_pixels * pixel_area_m2
        
        results = {
            'pcf_percent': float(pcf_percent),
            'pcf_fraction': float(pcf_fraction),
            'covered_pixels': int(covered_pixels),
            'total_pixels': int(total_pixels),
            'covered_area_m2': float(covered_area_m2),
            'total_area_m2': float(total_area_m2)
        }
        
        print(f"  → PCF: {pcf_percent:.2f}% ({covered_area_m2:.4f} m² / {total_area_m2:.4f} m²)")
        
        return results
    
    def calculate_abundance(self) -> Dict[str, float]:
        """
        Calculate Resource Density (Abundance) - The Economic Grade Metric.
        
        This metric counts the number of distinct objects per unit area,
        representing the spatial density of the resource. In deep-sea mining,
        this is a key indicator of economic viability.
        
        Formula:
            Abundance = (Number of polygons) / (Total area in m²)
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'count': Total number of objects
                - 'abundance_per_m2': Objects per square meter
                - 'abundance_per_100m2': Objects per 100 square meters
                - 'total_area_m2': Total area in square meters
        
        Raises:
            ValueError: If geojson_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(
            ...     geojson_path="nodules.geojson",
            ...     mask_path="mask.tif"
            ... )
            >>> abundance = analyzer.calculate_abundance()
            >>> print(f"Density: {abundance['abundance_per_m2']:.2f} objects/m²")
        
        Note:
            For mass-based density (kg/m²), volume estimation from depth maps
            would be required. This metric provides count-based density.
        """
        print("\n[METRIC] Calculating Resource Abundance...")
        
        # Load polygons
        polygons = self._load_polygons()
        count = len(polygons)
        
        # Get total area from mask or calculate from polygons
        if self.mask_path is not None:
            mask = self._load_mask()
            total_pixels = mask.size
            total_area_m2 = total_pixels * (self.meters_per_pixel ** 2)
        else:
            # Estimate from polygon bounding boxes
            if count > 0:
                all_bounds = [p.bounds for p in polygons]
                min_x = min(b[0] for b in all_bounds)
                min_y = min(b[1] for b in all_bounds)
                max_x = max(b[2] for b in all_bounds)
                max_y = max(b[3] for b in all_bounds)
                total_area_m2 = (max_x - min_x) * (max_y - min_y) * (self.meters_per_pixel ** 2)
            else:
                total_area_m2 = 1.0  # Avoid division by zero
        
        # Calculate abundance metrics
        abundance_per_m2 = count / total_area_m2 if total_area_m2 > 0 else 0
        abundance_per_100m2 = abundance_per_m2 * 100
        
        results = {
            'count': int(count),
            'abundance_per_m2': float(abundance_per_m2),
            'abundance_per_100m2': float(abundance_per_100m2),
            'total_area_m2': float(total_area_m2)
        }
        
        print(f"  → Count: {count} objects")
        print(f"  → Abundance: {abundance_per_m2:.4f} objects/m² ({abundance_per_100m2:.2f} per 100m²)")
        
        return results
    
    def calculate_spatial_homogeneity(self, grid_size: int = 4) -> Dict[str, float]:
        """
        Calculate Spatial Homogeneity via Quadrat Analysis - The Patchiness Metric.
        
        This metric quantifies how uniformly objects are distributed across the
        study area. It divides the area into a grid and calculates the variance
        of object counts across cells.
        
        The Variance-to-Mean Ratio (VMR), also known as the Index of Dispersion,
        indicates the distribution pattern:
            - VMR ≈ 1: Random (Poisson) distribution
            - VMR < 1: Uniform/regular distribution
            - VMR > 1: Clustered/aggregated distribution
        
        Formula:
            VMR = Variance(counts) / Mean(counts)
        
        Args:
            grid_size: Number of grid cells per dimension (default: 4).
                      Total cells = grid_size × grid_size.
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'vmr': Variance-to-Mean Ratio (Index of Dispersion)
                - 'mean_count': Mean objects per cell
                - 'variance': Variance of counts across cells
                - 'std_dev': Standard deviation of counts
                - 'min_count': Minimum objects in any cell
                - 'max_count': Maximum objects in any cell
                - 'pattern': Interpreted pattern ('clustered', 'random', 'uniform')
                - 'grid_size': Grid dimensions used
                - 'cell_counts': List of counts per cell
        
        Raises:
            ValueError: If neither mask_path nor geojson_path was provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(geojson_path="nodules.geojson")
            >>> homogeneity = analyzer.calculate_spatial_homogeneity(grid_size=5)
            >>> print(f"Pattern: {homogeneity['pattern']} (VMR={homogeneity['vmr']:.2f})")
        
        Note:
            Low variance indicates even distribution (homogeneous), while high
            variance indicates clustered distribution (patchy/heterogeneous).
        """
        print(f"\n[METRIC] Calculating Spatial Homogeneity (Quadrat Analysis, {grid_size}x{grid_size})...")
        
        # Get polygon centroids
        polygons = self._load_polygons()
        
        if len(polygons) == 0:
            print("[WARNING] No polygons found, returning default values")
            return {
                'vmr': 0.0,
                'mean_count': 0.0,
                'variance': 0.0,
                'std_dev': 0.0,
                'min_count': 0,
                'max_count': 0,
                'pattern': 'empty',
                'grid_size': grid_size,
                'cell_counts': []
            }
        
        # Get centroids
        centroids = np.array([[p.centroid.x, p.centroid.y] for p in polygons])
        
        # Determine bounds
        min_x, min_y = centroids.min(axis=0)
        max_x, max_y = centroids.max(axis=0)
        
        # Add small buffer to include edge points
        buffer = 0.001 * max(max_x - min_x, max_y - min_y)
        min_x -= buffer
        min_y -= buffer
        max_x += buffer
        max_y += buffer
        
        # Create grid
        x_edges = np.linspace(min_x, max_x, grid_size + 1)
        y_edges = np.linspace(min_y, max_y, grid_size + 1)
        
        # Count objects in each cell
        cell_counts = []
        for i in tqdm(range(grid_size), desc="Processing grid rows", unit="row"):
            for j in range(grid_size):
                # Define cell bounds
                x_min, x_max = x_edges[j], x_edges[j + 1]
                y_min, y_max = y_edges[i], y_edges[i + 1]
                
                # Count centroids in cell
                in_cell = (
                    (centroids[:, 0] >= x_min) & (centroids[:, 0] < x_max) &
                    (centroids[:, 1] >= y_min) & (centroids[:, 1] < y_max)
                )
                cell_counts.append(int(np.sum(in_cell)))
        
        # Calculate statistics
        cell_counts_arr = np.array(cell_counts)
        mean_count = np.mean(cell_counts_arr)
        variance = np.var(cell_counts_arr, ddof=1) if len(cell_counts_arr) > 1 else 0
        std_dev = np.std(cell_counts_arr, ddof=1) if len(cell_counts_arr) > 1 else 0
        
        # Calculate VMR (Index of Dispersion)
        # Handle edge case where mean is zero
        vmr = variance / mean_count if mean_count > 0 else 0
        
        # Interpret pattern
        if mean_count == 0:
            pattern = 'empty'
        elif vmr < 0.8:
            pattern = 'uniform'
        elif vmr > 1.2:
            pattern = 'clustered'
        else:
            pattern = 'random'
        
        results = {
            'vmr': float(vmr),
            'mean_count': float(mean_count),
            'variance': float(variance),
            'std_dev': float(std_dev),
            'min_count': int(cell_counts_arr.min()),
            'max_count': int(cell_counts_arr.max()),
            'pattern': pattern,
            'grid_size': grid_size,
            'cell_counts': cell_counts
        }
        
        print(f"  → VMR: {vmr:.3f}")
        print(f"  → Mean count per cell: {mean_count:.2f} ± {std_dev:.2f}")
        print(f"  → Range: [{cell_counts_arr.min()}, {cell_counts_arr.max()}]")
        print(f"  → Pattern interpretation: {pattern.upper()}")
        
        return results
    
    # =========================================================================
    # PROXIMITY & CLUSTERING METRICS
    # =========================================================================
    
    def calculate_nearest_neighbor_distance(
        self, 
        method: str = 'edge'
    ) -> Dict[str, float]:
        """
        Calculate Nearest Neighbor Distance (NND) - The Physical Gap Metric.
        
        This metric measures the distance from each object to its closest neighbor.
        Two methods are supported:
        
        - 'edge': Edge-to-edge distance (physical gap between object boundaries).
                  Critical for engineering applications like collector jamming prediction.
        - 'centroid': Center-to-center distance (faster but less accurate).
                      Suitable for general spatial analysis.
        
        Formula (Edge-to-Edge):
            NND_i = min(distance(boundary_i, boundary_j)) for all j ≠ i
        
        Formula (Centroid):
            NND_i = min(distance(centroid_i, centroid_j)) for all j ≠ i
        
        Args:
            method: Distance calculation method. Options:
                   - 'edge': Edge-to-edge using Shapely (more accurate, slower)
                   - 'centroid': Center-to-center using KDTree (faster)
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'method': Method used ('edge' or 'centroid')
                - 'mean_nnd_m': Mean nearest neighbor distance in meters
                - 'median_nnd_m': Median nearest neighbor distance in meters
                - 'std_nnd_m': Standard deviation in meters
                - 'min_nnd_m': Minimum distance in meters
                - 'max_nnd_m': Maximum distance in meters
                - 'nnd_values_m': List of all NND values in meters
        
        Raises:
            ValueError: If geojson_path was not provided or method is invalid.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(geojson_path="nodules.geojson")
            >>> # Edge-to-edge for engineering analysis
            >>> nnd_edge = analyzer.calculate_nearest_neighbor_distance(method='edge')
            >>> print(f"Mean gap: {nnd_edge['mean_nnd_m']:.3f} m")
            >>> 
            >>> # Centroid for faster analysis
            >>> nnd_centroid = analyzer.calculate_nearest_neighbor_distance(method='centroid')
        
        Note:
            For engineering applications (jamming prediction, collision analysis),
            use 'edge' method. Two objects might have centroids 15cm apart, but
            edges only 3cm apart - a critical difference for collector design.
        """
        print(f"\n[METRIC] Calculating Nearest Neighbor Distance (method={method})...")
        
        if method not in ['edge', 'centroid']:
            raise ValueError(f"Invalid method '{method}'. Use 'edge' or 'centroid'.")
        
        # Load polygons
        polygons = self._load_polygons()
        n = len(polygons)
        
        if n < 2:
            print("[WARNING] Need at least 2 polygons for NND calculation")
            return {
                'method': method,
                'mean_nnd_m': 0.0,
                'median_nnd_m': 0.0,
                'std_nnd_m': 0.0,
                'min_nnd_m': 0.0,
                'max_nnd_m': 0.0,
                'nnd_values_m': []
            }
        
        nnd_values = []
        
        if method == 'centroid':
            # Fast method using KDTree on centroids
            centroids = np.array([[p.centroid.x, p.centroid.y] for p in polygons])
            tree = KDTree(centroids)
            
            # Query for 2 nearest neighbors (first is self, second is nearest other)
            distances, _ = tree.query(centroids, k=2)
            nnd_values = distances[:, 1].tolist()  # Second column is nearest neighbor
            
        else:  # method == 'edge'
            # Accurate method using Shapely distance (edge-to-edge)
            for i in tqdm(range(n), desc="Computing edge distances", unit="polygon"):
                min_dist = float('inf')
                
                for j in range(n):
                    if i == j:
                        continue
                    
                    # Shapely .distance() returns minimum edge-to-edge distance
                    dist = polygons[i].distance(polygons[j])
                    if dist < min_dist:
                        min_dist = dist
                
                nnd_values.append(min_dist)
        
        # Convert to meters
        nnd_values_m = [d * self.meters_per_pixel for d in nnd_values]
        nnd_arr = np.array(nnd_values_m)
        
        results = {
            'method': method,
            'mean_nnd_m': float(np.mean(nnd_arr)),
            'median_nnd_m': float(np.median(nnd_arr)),
            'std_nnd_m': float(np.std(nnd_arr)),
            'min_nnd_m': float(np.min(nnd_arr)),
            'max_nnd_m': float(np.max(nnd_arr)),
            'nnd_values_m': nnd_values_m
        }
        
        print(f"  → Mean NND: {results['mean_nnd_m']:.4f} m")
        print(f"  → Median NND: {results['median_nnd_m']:.4f} m")
        print(f"  → Range: [{results['min_nnd_m']:.4f}, {results['max_nnd_m']:.4f}] m")
        
        return results
    
    def calculate_passability_index(self) -> Dict[str, float]:
        """
        Calculate Passability Index - The Corridor/Navigable Space Metric.
        
        This metric analyzes the "voids" (empty space) between objects to determine
        the maximum size of a vehicle or tool that can navigate through the field
        without colliding with objects.
        
        The calculation uses a Euclidean Distance Transform (EDT) on the inverted
        binary mask to find the maximum inscribed circle radius.
        
        Formula:
            Passability = 2 × max(EDT(inverted_mask)) × meters_per_pixel
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'max_passage_diameter_m': Diameter of largest clear passage (meters)
                - 'max_passage_radius_m': Radius of largest clear passage (meters)
                - 'mean_clearance_m': Mean distance to nearest obstacle (meters)
                - 'median_clearance_m': Median clearance (meters)
                - 'passability_fraction': Fraction of area that is passable
        
        Raises:
            ValueError: If mask_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(mask_path="obstacles.tif")
            >>> passability = analyzer.calculate_passability_index()
            >>> print(f"Max vehicle width: {passability['max_passage_diameter_m']:.2f} m")
        
        Note:
            If outcrops are scattered randomly, passability is typically high.
            If outcrops form linear ridges, passability can be very low.
            This metric is critical for path planning in mining operations.
        """
        print("\n[METRIC] Calculating Passability Index...")
        
        # Load mask data
        mask = self._load_mask()
        
        # Define free space and obstacles based on semantic mask
        # Mask structure: 0=background, 1=substrate, 2=nodule, 3=organisms
        # Free space is substrate (mask == 1)
        # Obstacles are nodules and organisms (mask >= 2)
        free_space = (mask == 1).astype(np.float32)
        
        # Check if there's any free space
        if np.sum(free_space) == 0:
            print("[WARNING] No free space found in mask")
            return {
                'max_passage_diameter_m': 0.0,
                'max_passage_radius_m': 0.0,
                'mean_clearance_m': 0.0,
                'median_clearance_m': 0.0,
                'passability_fraction': 0.0
            }
        
        # Check if there are any obstacles (nodules or organisms)
        if np.sum(mask >= 2) == 0:
            print("[WARNING] No obstacles found in mask - infinite passability")
            diagonal = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2) * self.meters_per_pixel
            return {
                'max_passage_diameter_m': diagonal,
                'max_passage_radius_m': diagonal / 2,
                'mean_clearance_m': diagonal / 2,
                'median_clearance_m': diagonal / 2,
                'passability_fraction': 1.0
            }
        
        print("  Computing Euclidean Distance Transform...")
        # EDT gives distance from each free pixel to nearest obstacle
        edt = ndimage.distance_transform_edt(free_space)
        
        # Find maximum radius (the largest inscribed circle) and its location
        max_radius_px = np.max(edt)
        max_idx = np.unravel_index(np.argmax(edt), edt.shape)
        max_pixel_row, max_pixel_col = max_idx
        
        # Convert pixel coordinates to world coordinates (meters)
        max_world_x = max_pixel_col * self.meters_per_pixel
        max_world_y = max_pixel_row * self.meters_per_pixel
        
        max_radius_m = max_radius_px * self.meters_per_pixel
        max_diameter_m = 2 * max_radius_m
        
        # Calculate mean and median clearance in free space only
        free_space_edt = edt[free_space > 0]
        mean_clearance_m = np.mean(free_space_edt) * self.meters_per_pixel
        median_clearance_m = np.median(free_space_edt) * self.meters_per_pixel
        
        # Passability fraction (free space / total)
        passability_fraction = np.sum(free_space) / mask.size
        
        results = {
            'max_passage_diameter_m': float(max_diameter_m),
            'max_passage_radius_m': float(max_radius_m),
            'max_passage_x_m': float(max_world_x),
            'max_passage_y_m': float(max_world_y),
            'max_passage_pixel_row': int(max_pixel_row),
            'max_passage_pixel_col': int(max_pixel_col),
            'mean_clearance_m': float(mean_clearance_m),
            'median_clearance_m': float(median_clearance_m),
            'passability_fraction': float(passability_fraction)
        }
        
        print(f"  → Max passage diameter: {max_diameter_m:.4f} m")
        print(f"  → Max passage location: ({max_world_x:.4f}, {max_world_y:.4f}) m")
        print(f"  → Mean clearance: {mean_clearance_m:.4f} m")
        print(f"  → Free space: {passability_fraction * 100:.2f}%")
        
        return results
    
    def calculate_ripleys_k(
        self, 
        radii: Optional[List[float]] = None,
        n_radii: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate Ripley's K Function - The Scale of Aggregation Metric.
        
        Ripley's K is a multi-scale measure of spatial clustering that reveals
        at what scale(s) objects are clustered. Unlike simple density metrics,
        it distinguishes between "small tight groups" and "large loose patches."
        
        The K function counts the average number of neighbors within distance r
        and compares this to a random (Poisson) distribution:
            - K(r) > expected: Clustering at scale r
            - K(r) ≈ expected: Random distribution at scale r
            - K(r) < expected: Dispersion/regularity at scale r
        
        The L function (normalized K) is also computed:
            L(r) = sqrt(K(r)/π) - r
        
        Formula:
            K(r) = (A/n²) × Σᵢ Σⱼ I(dᵢⱼ ≤ r), where I is indicator function
        
        Args:
            radii: List of radii (in meters) to evaluate. If None, automatically
                  generated from min to max inter-point distance.
            n_radii: Number of radii to evaluate if radii is None (default: 20).
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'radii_m': List of evaluated radii in meters
                - 'k_observed': Observed K values at each radius
                - 'k_expected': Expected K values under random distribution
                - 'l_function': Normalized L(r) = sqrt(K/π) - r values
                - 'clustering_peaks': Radii where significant clustering detected
                - 'max_clustering_radius_m': Radius with maximum clustering
        
        Raises:
            ValueError: If geojson_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(geojson_path="nodules.geojson")
            >>> ripley = analyzer.calculate_ripleys_k(n_radii=30)
            >>> print(f"Peak clustering at: {ripley['max_clustering_radius_m']:.2f} m")
        
        Note:
            A positive peak at r=2m indicates nodules arranged in patches roughly
            2m wide. Multiple peaks indicate multi-scale clustering (e.g., small
            clusters within larger clusters).
        """
        print("\n[METRIC] Calculating Ripley's K Function...")
        
        # Load polygons and get centroids
        polygons = self._load_polygons()
        n = len(polygons)
        
        if n < 3:
            print("[WARNING] Need at least 3 points for Ripley's K")
            return {
                'radii_m': [],
                'k_observed': [],
                'k_expected': [],
                'l_function': [],
                'clustering_peaks': [],
                'max_clustering_radius_m': 0.0
            }
        
        # Get centroids in physical coordinates (meters)
        centroids = np.array([
            [p.centroid.x * self.meters_per_pixel, 
             p.centroid.y * self.meters_per_pixel] 
            for p in polygons
        ])
        
        # Calculate study area
        min_x, min_y = centroids.min(axis=0)
        max_x, max_y = centroids.max(axis=0)
        area = (max_x - min_x) * (max_y - min_y)
        
        if area == 0:
            print("[WARNING] Zero area - all points at same location")
            return {
                'radii_m': [],
                'k_observed': [],
                'k_expected': [],
                'l_function': [],
                'clustering_peaks': [],
                'max_clustering_radius_m': 0.0
            }
        
        # Generate radii if not provided
        if radii is None:
            # Use inter-point distances to determine range
            tree = KDTree(centroids)
            distances, _ = tree.query(centroids, k=min(n, 10))
            max_dist = np.max(distances)
            radii = np.linspace(max_dist * 0.05, max_dist * 0.5, n_radii).tolist()
        
        # Build KDTree for efficient neighbor counting
        tree = KDTree(centroids)
        
        # Calculate K(r) for each radius
        k_observed = []
        k_expected = []
        l_function = []
        
        for r in tqdm(radii, desc="Computing K(r)", unit="radius"):
            # Count pairs within distance r
            # query_ball_point returns indices of points within distance r
            pair_count = 0
            for i in range(n):
                neighbors = tree.query_ball_point(centroids[i], r)
                # Subtract 1 to exclude self
                pair_count += len(neighbors) - 1
            
            # Observed K(r) = (Area / n²) × pair_count
            k_obs = (area / (n * n)) * pair_count
            k_observed.append(k_obs)
            
            # Expected K(r) for random Poisson process = π × r²
            k_exp = np.pi * r * r
            k_expected.append(k_exp)
            
            # L function: L(r) = sqrt(K(r)/π) - r
            # Positive L indicates clustering, negative indicates dispersion
            l_val = np.sqrt(k_obs / np.pi) - r if k_obs > 0 else -r
            l_function.append(l_val)
        
        # Find clustering peaks (where L > 0 significantly)
        l_arr = np.array(l_function)
        threshold = np.std(l_arr) * 0.5 if len(l_arr) > 0 else 0
        clustering_peaks = [r for r, l in zip(radii, l_function) if l > threshold]
        
        # Find radius of maximum clustering
        max_idx = np.argmax(l_arr) if len(l_arr) > 0 else 0
        max_clustering_radius = radii[max_idx] if len(radii) > 0 else 0.0
        
        results = {
            'radii_m': radii,
            'k_observed': k_observed,
            'k_expected': k_expected,
            'l_function': l_function,
            'clustering_peaks': clustering_peaks,
            'max_clustering_radius_m': float(max_clustering_radius)
        }
        
        print(f"  → Evaluated {len(radii)} radii from {min(radii):.3f} to {max(radii):.3f} m")
        print(f"  → Max clustering at radius: {max_clustering_radius:.4f} m")
        print(f"  → Found {len(clustering_peaks)} significant clustering peaks")
        
        return results
    
    def calculate_clark_evans(self) -> Dict[str, float]:
        """
        Calculate Clark-Evans Aggregation Index - The Clustering Fingerprint Metric.
        
        The Clark-Evans Index (R) is a single number that summarizes the clustering
        intensity of a point pattern. It compares the observed mean nearest neighbor
        distance to the expected distance under complete spatial randomness.
        
        Formula:
            R = r̄_observed / r̄_expected
            r̄_expected = 1 / (2 × sqrt(density))
            density = n / Area
        
        Interpretation:
            - R = 1.0: Random (Poisson) distribution
            - R > 1.0: Uniform/dispersed (objects further apart than random)
            - R < 1.0: Clustered (objects closer together than random)
            - R → 0.0: Maximum clustering (all points at same location)
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'clark_evans_r': The R index value
                - 'pattern': Interpreted pattern ('clustered', 'random', 'dispersed')
                - 'mean_observed_distance_m': Observed mean NND in meters
                - 'mean_expected_distance_m': Expected mean NND in meters
                - 'density_per_m2': Point density (objects per m²)
                - 'z_score': Statistical significance (standard deviations from random)
        
        Raises:
            ValueError: If geojson_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(geojson_path="nodules.geojson")
            >>> ce = analyzer.calculate_clark_evans()
            >>> print(f"Clark-Evans R: {ce['clark_evans_r']:.3f} ({ce['pattern']})")
        
        Note:
            While faster than Ripley's K, the Clark-Evans Index provides less detail.
            Use it for quick screening; use Ripley's K for detailed scale analysis.
            R < 1 is like saying "neighbors are closer than they should be" - clustered.
        """
        print("\n[METRIC] Calculating Clark-Evans Aggregation Index...")
        
        # Load polygons and get centroids
        polygons = self._load_polygons()
        n = len(polygons)
        
        if n < 2:
            print("[WARNING] Need at least 2 points for Clark-Evans")
            return {
                'clark_evans_r': 1.0,
                'pattern': 'insufficient_data',
                'mean_observed_distance_m': 0.0,
                'mean_expected_distance_m': 0.0,
                'density_per_m2': 0.0,
                'z_score': 0.0
            }
        
        # Get centroids in physical coordinates
        centroids = np.array([
            [p.centroid.x * self.meters_per_pixel, 
             p.centroid.y * self.meters_per_pixel] 
            for p in polygons
        ])
        
        # Calculate study area
        min_x, min_y = centroids.min(axis=0)
        max_x, max_y = centroids.max(axis=0)
        
        # Add buffer to avoid edge effects
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        
        if area == 0:
            # All points at same x or y - use minimum bounding
            area = max(width, height) ** 2 if max(width, height) > 0 else 1.0
        
        # Calculate density
        density = n / area
        
        # Calculate observed mean nearest neighbor distance
        tree = KDTree(centroids)
        distances, _ = tree.query(centroids, k=2)
        mean_observed = np.mean(distances[:, 1])
        
        # Calculate expected mean NND for random distribution
        # r̄_expected = 1 / (2 × sqrt(density)) = 0.5 / sqrt(n/A) = 0.5 × sqrt(A/n)
        mean_expected = 0.5 / np.sqrt(density) if density > 0 else float('inf')
        
        # Clark-Evans R statistic
        r_index = mean_observed / mean_expected if mean_expected > 0 else 0
        
        # Calculate z-score for statistical significance
        # Standard error = 0.26136 / sqrt(n × density)
        se = 0.26136 / np.sqrt(n * density) if density > 0 else 1
        z_score = (mean_observed - mean_expected) / se if se > 0 else 0
        
        # Interpret pattern
        if r_index < 0.8:
            pattern = 'clustered'
        elif r_index > 1.2:
            pattern = 'dispersed'
        else:
            pattern = 'random'
        
        results = {
            'clark_evans_r': float(r_index),
            'pattern': pattern,
            'mean_observed_distance_m': float(mean_observed),
            'mean_expected_distance_m': float(mean_expected),
            'density_per_m2': float(density),
            'z_score': float(z_score)
        }
        
        print(f"  → Clark-Evans R: {r_index:.4f}")
        print(f"  → Pattern: {pattern.upper()}")
        print(f"  → Observed mean NND: {mean_observed:.4f} m")
        print(f"  → Expected mean NND: {mean_expected:.4f} m")
        print(f"  → Z-score: {z_score:.2f}")
        
        return results
    
    # =========================================================================
    # INDIVIDUAL MORPHOLOGY METRICS
    # =========================================================================
    
    def calculate_morphology_stats(self) -> pd.DataFrame:
        """
        Calculate Individual Morphology Statistics - Shape Characterization.
        
        This method computes shape descriptors for each polygon, characterizing
        individual object morphology. Three key metrics are calculated:
        
        1. **Circularity** (Isoperimetric Quotient): Measures roundness.
           Formula: 4π × Area / Perimeter²
           - 1.0 = Perfect circle
           - 0.7-0.9 = Ellipsoid (potato-shaped nodule)
           - < 0.5 = Irregular (starfish, jagged rock)
        
        2. **Solidity** (Convexity): Measures smoothness/jaggedness.
           Formula: Area / Convex_Hull_Area
           - 1.0 = Perfectly convex (smooth egg shape)
           - Low = Deep cavities or jagged protrusions
        
        3. **OBB Aspect Ratio**: Measures elongation via Oriented Bounding Box.
           Formula: min(width, height) / max(width, height)
           - 1.0 = Square/circular (equidimensional)
           - → 0 = Long and skinny (elongated)
        
        Returns:
            pd.DataFrame: DataFrame with one row per polygon containing:
                - 'polygon_id': Unique identifier (0-indexed)
                - 'area_m2': Polygon area in square meters
                - 'perimeter_m': Polygon perimeter in meters
                - 'circularity': Isoperimetric quotient (0-1)
                - 'solidity': Convexity ratio (0-1)
                - 'obb_aspect_ratio': OBB width/length ratio (0-1)
                - 'centroid_x_m': X coordinate of centroid in meters
                - 'centroid_y_m': Y coordinate of centroid in meters
        
        Raises:
            ValueError: If geojson_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(geojson_path="nodules.geojson")
            >>> morphology = analyzer.calculate_morphology_stats()
            >>> print(f"Mean circularity: {morphology['circularity'].mean():.3f}")
            >>> # Filter for round nodules
            >>> round_nodules = morphology[morphology['circularity'] > 0.8]
        
        Note:
            These metrics assume orthorectified polygons. For oblique imagery,
            circularity will be artificially low due to perspective distortion.
        """
        print("\n[METRIC] Calculating Morphology Statistics...")
        
        # Load polygons
        polygons = self._load_polygons()
        
        if len(polygons) == 0:
            print("[WARNING] No polygons found")
            return pd.DataFrame()
        
        # Prepare results storage
        results = []
        
        for i, poly in enumerate(tqdm(polygons, desc="Analyzing shapes", unit="polygon")):
            # Convert coordinates to meters
            scale = self.meters_per_pixel
            
            # Basic measurements (already in coordinate units, scale to meters)
            area = poly.area * (scale ** 2)
            perimeter = poly.length * scale
            
            # -----------------------------------------------------------------
            # CIRCULARITY (Isoperimetric Quotient)
            # Formula: 4π × Area / Perimeter²
            # Perfect circle = 1.0, irregular shapes < 1.0
            # -----------------------------------------------------------------
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                # Clamp to [0, 1] to handle floating point errors
                circularity = min(1.0, max(0.0, circularity))
            else:
                circularity = 0.0
            
            # -----------------------------------------------------------------
            # SOLIDITY (Convexity / Rugosity Proxy)
            # Formula: Area / Convex_Hull_Area
            # Smooth convex shape = 1.0, jagged/concave < 1.0
            # -----------------------------------------------------------------
            convex_hull = poly.convex_hull
            hull_area = convex_hull.area * (scale ** 2)
            
            if hull_area > 0:
                solidity = area / hull_area
                solidity = min(1.0, max(0.0, solidity))
            else:
                solidity = 0.0
            
            # -----------------------------------------------------------------
            # OBB ASPECT RATIO (Oriented Bounding Box)
            # Uses minimum rotated rectangle to find principal axis
            # Formula: min(width, height) / max(width, height)
            # Square/circle ≈ 1.0, elongated → 0.0
            # -----------------------------------------------------------------
            try:
                # minimum_rotated_rectangle returns the OBB as a polygon
                obb = poly.minimum_rotated_rectangle
                obb_coords = list(obb.exterior.coords)
                
                # Calculate side lengths of the OBB
                # The rectangle has 5 coords (closed ring), use first 3 for 2 sides
                side1 = np.sqrt(
                    (obb_coords[1][0] - obb_coords[0][0])**2 + 
                    (obb_coords[1][1] - obb_coords[0][1])**2
                ) * scale
                side2 = np.sqrt(
                    (obb_coords[2][0] - obb_coords[1][0])**2 + 
                    (obb_coords[2][1] - obb_coords[1][1])**2
                ) * scale
                
                # Aspect ratio = short side / long side
                if max(side1, side2) > 0:
                    obb_aspect_ratio = min(side1, side2) / max(side1, side2)
                else:
                    obb_aspect_ratio = 1.0
            except Exception:
                # Fallback for degenerate polygons
                obb_aspect_ratio = 1.0
            
            # -----------------------------------------------------------------
            # CENTROID LOCATION
            # -----------------------------------------------------------------
            centroid_x_m = poly.centroid.x * scale
            centroid_y_m = poly.centroid.y * scale
            
            # Store results
            results.append({
                'polygon_id': i,
                'area_m2': area,
                'perimeter_m': perimeter,
                'circularity': circularity,
                'solidity': solidity,
                'obb_aspect_ratio': obb_aspect_ratio,
                'centroid_x_m': centroid_x_m,
                'centroid_y_m': centroid_y_m
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Print summary statistics
        print(f"  → Analyzed {len(df)} polygons")
        print(f"  → Circularity: mean={df['circularity'].mean():.3f}, "
              f"std={df['circularity'].std():.3f}")
        print(f"  → Solidity: mean={df['solidity'].mean():.3f}, "
              f"std={df['solidity'].std():.3f}")
        print(f"  → OBB Aspect Ratio: mean={df['obb_aspect_ratio'].mean():.3f}, "
              f"std={df['obb_aspect_ratio'].std():.3f}")
        
        return df
    
    # =========================================================================
    # VERTICALITY & INTERACTION METRICS (3D - Require Elevation)
    # =========================================================================
    
    def calculate_protrusion(self) -> Dict[str, Any]:
        """
        Calculate Protrusion (Stick-up Height) - The Collector Clearance Metric.
        
        This metric measures how far objects protrude above the local seafloor,
        which is critical for mining vehicle collector head settings. Objects
        that "stick up" more are easier to collect without disturbing sediment.
        
        Method:
            1. Load elevation raster and mask
            2. Identify sediment pixels (mask == 0)
            3. Interpolate a "virtual seafloor plane" using RANSAC or mean
            4. Calculate height of object pixels above this plane
            5. Return statistics (mean, max, distribution)
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'mean_protrusion_m': Mean stick-up height in meters
                - 'median_protrusion_m': Median stick-up height in meters
                - 'max_protrusion_m': Maximum stick-up height in meters
                - 'std_protrusion_m': Standard deviation in meters
                - 'seafloor_elevation_m': Estimated seafloor plane elevation
                - 'protrusion_histogram': Histogram of protrusion values
                - 'per_object_stats': Per-polygon protrusion statistics (if polygons available)
        
        Raises:
            ValueError: If elevation_path or mask_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(
            ...     elevation_path="depth.tif",
            ...     mask_path="nodules.tif"
            ... )
            >>> protrusion = analyzer.calculate_protrusion()
            >>> print(f"Mean stick-up: {protrusion['mean_protrusion_m']*100:.1f} cm")
        
        Note:
            If mean stick-up is 2cm, the collector must be lowered aggressively.
            If stick-up is 10cm, the collector can "clip" nodules off the top.
            Filter feeders may prefer high stick-up nodules for current access.
        """
        print("\n[METRIC] Calculating Protrusion (Stick-up Height)...")
        
        # Load required data
        elevation = self._load_elevation()
        mask = self._load_mask()
        
        # Resample elevation to match mask dimensions if needed
        # (They may cover the same geographic area at different resolutions)
        elevation = self._resample_elevation_to_mask(elevation, mask)
        
        # Identify sediment (substrate) and object pixels
        # Mask structure: 0=background, 1=substrate, 2=nodule, 3=organisms
        # Substrate (mask == 1) is the reference seafloor elevation
        # Objects (mask >= 2) are nodules and organisms
        sediment_mask = (mask == 1) & (~np.isnan(elevation))
        object_mask = (mask >= 2) & (~np.isnan(elevation))
        
        if not np.any(sediment_mask):
            print("[WARNING] No sediment pixels found for seafloor estimation")
            return {
                'mean_protrusion_m': 0.0,
                'median_protrusion_m': 0.0,
                'max_protrusion_m': 0.0,
                'std_protrusion_m': 0.0,
                'seafloor_elevation_m': 0.0,
                'protrusion_histogram': {},
                'per_object_stats': []
            }
        
        if not np.any(object_mask):
            print("[WARNING] No object pixels found in mask")
            return {
                'mean_protrusion_m': 0.0,
                'median_protrusion_m': 0.0,
                'max_protrusion_m': 0.0,
                'std_protrusion_m': 0.0,
                'seafloor_elevation_m': float(np.nanmean(elevation[sediment_mask])),
                'protrusion_histogram': {},
                'per_object_stats': []
            }
        
        # -----------------------------------------------------------------
        # ESTIMATE SEAFLOOR PLANE
        # Using robust mean of sediment pixels (simple approach)
        # For more accuracy, RANSAC plane fitting could be used
        # -----------------------------------------------------------------
        print("  Estimating virtual seafloor plane...")
        sediment_elevations = elevation[sediment_mask]
        
        # Use median for robustness against outliers
        seafloor_elevation = np.median(sediment_elevations)
        
        # -----------------------------------------------------------------
        # CALCULATE PROTRUSION
        # Height above seafloor (positive = sticking up)
        # -----------------------------------------------------------------
        object_elevations = elevation[object_mask]
        protrusion_values = object_elevations - seafloor_elevation
        
        # Convert to meters (assuming elevation is already in correct units)
        # If elevation is in image units, scale by meters_per_pixel for Z
        # Note: This assumes elevation values are in the same unit as XY scale
        
        # Calculate statistics
        mean_protrusion = float(np.mean(protrusion_values))
        median_protrusion = float(np.median(protrusion_values))
        max_protrusion = float(np.max(protrusion_values))
        std_protrusion = float(np.std(protrusion_values))
        
        # Create histogram
        hist, bin_edges = np.histogram(protrusion_values, bins=20)
        protrusion_histogram = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        # -----------------------------------------------------------------
        # PER-OBJECT STATISTICS (if polygons available)
        # -----------------------------------------------------------------
        per_object_stats = []
        if self.geojson_path is not None:
            try:
                polygons = self._load_polygons()
                print(f"  Calculating per-object protrusion for {len(polygons)} objects...")
                
                # We need raster coordinates for polygon operations
                # This is a simplified approach - full implementation would
                # rasterize each polygon and extract elevation values
                
                for i, poly in enumerate(tqdm(polygons[:100], desc="Per-object stats", unit="obj")):
                    # Get bounding box in raster coordinates
                    minx, miny, maxx, maxy = poly.bounds
                    
                    # Convert to pixel coordinates (approximate)
                    row_min = int(max(0, miny))
                    row_max = int(min(elevation.shape[0], maxy + 1))
                    col_min = int(max(0, minx))
                    col_max = int(min(elevation.shape[1], maxx + 1))
                    
                    # Extract region
                    if row_max > row_min and col_max > col_min:
                        region_elev = elevation[row_min:row_max, col_min:col_max]
                        region_mask = mask[row_min:row_max, col_min:col_max]
                        
                        obj_pixels = region_elev[region_mask > 0]
                        if len(obj_pixels) > 0:
                            obj_protrusion = obj_pixels - seafloor_elevation
                            per_object_stats.append({
                                'polygon_id': i,
                                'mean_protrusion': float(np.mean(obj_protrusion)),
                                'max_protrusion': float(np.max(obj_protrusion))
                            })
                            
            except Exception as e:
                print(f"[WARNING] Could not compute per-object stats: {e}")
        
        results = {
            'mean_protrusion_m': mean_protrusion,
            'median_protrusion_m': median_protrusion,
            'max_protrusion_m': max_protrusion,
            'std_protrusion_m': std_protrusion,
            'seafloor_elevation_m': float(seafloor_elevation),
            'protrusion_histogram': protrusion_histogram,
            'per_object_stats': per_object_stats
        }
        
        print(f"  → Mean protrusion: {mean_protrusion:.4f} m ({mean_protrusion*100:.2f} cm)")
        print(f"  → Max protrusion: {max_protrusion:.4f} m ({max_protrusion*100:.2f} cm)")
        print(f"  → Seafloor baseline: {seafloor_elevation:.4f} m")
        
        return results
    
    def calculate_3d_rugosity(self) -> Dict[str, float]:
        """
        Calculate 3D Surface Rugosity - The Texture Complexity Metric.
        
        This metric quantifies the surface texture complexity by comparing the
        true 3D surface area to the projected 2D footprint. Rough/textured
        surfaces have more surface area than smooth ones with the same footprint.
        
        Formula:
            Rugosity = True_3D_Surface_Area / Projected_2D_Area
        
        Method:
            1. Calculate gradient of elevation in X and Y directions
            2. Compute surface area element: sqrt(1 + (dz/dx)² + (dz/dy)²)
            3. Integrate over the object mask
            4. Divide by 2D pixel count
        
        Interpretation:
            - 1.0: Perfectly flat surface (theoretical minimum)
            - 1.0-1.2: Smooth surface (river stone)
            - 1.2-1.5: Moderate texture (typical nodule)
            - > 1.5: Highly complex surface (cauliflower, coral)
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'rugosity_index': Ratio of 3D to 2D area (≥1.0)
                - 'mean_slope_degrees': Mean surface slope in degrees
                - 'max_slope_degrees': Maximum surface slope in degrees
                - 'surface_area_3d_m2': Total 3D surface area in m²
                - 'surface_area_2d_m2': Projected 2D area in m²
        
        Raises:
            ValueError: If elevation_path or mask_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(
            ...     elevation_path="depth.tif",
            ...     mask_path="nodules.tif"
            ... )
            >>> rugosity = analyzer.calculate_3d_rugosity()
            >>> print(f"Surface complexity: {rugosity['rugosity_index']:.2f}x flat")
        
        Note:
            High rugosity nodules trap more sediment in their crevices, which
            increases the load on cleaning equipment. Combined with low protrusion,
            high rugosity indicates the "worst case" for sediment handling.
        """
        print("\n[METRIC] Calculating 3D Surface Rugosity...")
        
        # Load required data
        elevation = self._load_elevation()
        mask = self._load_mask()
        
        # Ensure same shape
        if elevation.shape != mask.shape:
            raise ValueError(
                f"Elevation shape {elevation.shape} does not match mask shape {mask.shape}"
            )
        
        # Get object pixels
        object_mask = (mask > 0) & (~np.isnan(elevation))
        
        if not np.any(object_mask):
            print("[WARNING] No valid object pixels found")
            return {
                'rugosity_index': 1.0,
                'mean_slope_degrees': 0.0,
                'max_slope_degrees': 0.0,
                'surface_area_3d_m2': 0.0,
                'surface_area_2d_m2': 0.0
            }
        
        # -----------------------------------------------------------------
        # CALCULATE GRADIENTS
        # Using numpy.gradient for partial derivatives
        # -----------------------------------------------------------------
        print("  Computing elevation gradients...")
        
        # Calculate gradient (change in elevation per pixel)
        # Scale by meters_per_pixel to get proper slope
        dz_dy, dz_dx = np.gradient(elevation)
        
        # -----------------------------------------------------------------
        # CALCULATE SURFACE AREA ELEMENT
        # For each pixel: dA = sqrt(1 + (dz/dx)² + (dz/dy)²) × pixel_area
        # This is the standard formula for surface area from a height field
        # -----------------------------------------------------------------
        
        # Surface area multiplier (local surface area / projected area)
        surface_multiplier = np.sqrt(1 + dz_dx**2 + dz_dy**2)
        
        # Calculate slope in degrees
        slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_degrees = np.degrees(slope_radians)
        
        # -----------------------------------------------------------------
        # INTEGRATE OVER OBJECT MASK
        # -----------------------------------------------------------------
        pixel_area_m2 = self.meters_per_pixel ** 2
        
        # 2D area (projected footprint)
        n_pixels = np.sum(object_mask)
        area_2d_m2 = n_pixels * pixel_area_m2
        
        # 3D area (true surface area)
        surface_elements = surface_multiplier[object_mask]
        area_3d_m2 = np.sum(surface_elements) * pixel_area_m2
        
        # Rugosity index
        rugosity_index = area_3d_m2 / area_2d_m2 if area_2d_m2 > 0 else 1.0
        
        # Slope statistics (only for object pixels)
        object_slopes = slope_degrees[object_mask]
        mean_slope = float(np.mean(object_slopes))
        max_slope = float(np.max(object_slopes))
        
        results = {
            'rugosity_index': float(rugosity_index),
            'mean_slope_degrees': mean_slope,
            'max_slope_degrees': max_slope,
            'surface_area_3d_m2': float(area_3d_m2),
            'surface_area_2d_m2': float(area_2d_m2)
        }
        
        print(f"  → Rugosity index: {rugosity_index:.4f}")
        print(f"  → Mean slope: {mean_slope:.2f}°")
        print(f"  → Max slope: {max_slope:.2f}°")
        print(f"  → 3D/2D area ratio: {area_3d_m2:.4f} / {area_2d_m2:.4f} m²")
        
        return results
    
    def calculate_exposed_volume(self) -> Dict[str, Any]:
        """
        Calculate Exposed Volume - The Yield Metric.
        
        This metric calculates the true physical volume protruding above the
        seafloor baseline by integrating stick-up heights across all object pixels.
        Unlike simplified sphere assumptions, this accounts for actual 3D shape.
        
        Method:
            1. Establish virtual seafloor plane (median of sediment pixels)
            2. Calculate stick-up height for each object pixel
            3. Integrate: Volume = Σ(height_i × pixel_area)
            4. Compute statistics and per-object volumes
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'total_exposed_volume_m3': Total volume of exposed nodules
                - 'mean_volume_per_object_m3': Average volume per polygon
                - 'volume_distribution': Histogram of volumes (20 bins)
                - 'per_object_volumes': List of volumes per polygon
                - 'num_exposed_pixels': Count of pixels protruding above seafloor
                - 'seafloor_elevation_m': Estimated baseline elevation
        
        Raises:
            ValueError: If elevation_path or mask_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(
            ...     elevation_path="depth.tif",
            ...     mask_path="nodules.tif"
            ... )
            >>> volume = analyzer.calculate_exposed_volume()
            >>> print(f"Total yield: {volume['total_exposed_volume_m3']:.2f} m³")
        """
        print("\n[METRIC] Calculating Exposed Volume...")
        
        # Load required data
        elevation = self._load_elevation()
        mask = self._load_mask()
        
        # Ensure same shape
        if elevation.shape != mask.shape:
            raise ValueError(
                f"Elevation shape {elevation.shape} does not match mask shape {mask.shape}"
            )
        
        # Identify sediment and object pixels (exclude NaN)
        # Mask structure: 0=background, 1=substrate, 2=nodule, 3=organisms
        # Substrate (mask == 1) is the reference seafloor elevation
        # Objects (mask >= 2) are nodules and organisms
        sediment_mask = (mask == 1) & (~np.isnan(elevation))
        object_mask = (mask >= 2) & (~np.isnan(elevation))
        
        if not np.any(sediment_mask):
            print("[WARNING] No sediment pixels found for seafloor estimation")
            return {
                'total_exposed_volume_m3': 0.0,
                'mean_volume_per_object_m3': 0.0,
                'volume_distribution': {'counts': [], 'bin_edges': []},
                'per_object_volumes': [],
                'num_exposed_pixels': 0,
                'seafloor_elevation_m': 0.0
            }
        
        if not np.any(object_mask):
            print("[WARNING] No object pixels found in mask")
            return {
                'total_exposed_volume_m3': 0.0,
                'mean_volume_per_object_m3': 0.0,
                'volume_distribution': {'counts': [], 'bin_edges': []},
                'per_object_volumes': [],
                'num_exposed_pixels': 0,
                'seafloor_elevation_m': float(np.nanmedian(elevation[sediment_mask]))
            }
        
        # -----------------------------------------------------------------
        # ESTIMATE SEAFLOOR PLANE (median of sediment pixels)
        # -----------------------------------------------------------------
        print("  Estimating virtual seafloor plane...")
        sediment_elevations = elevation[sediment_mask]
        seafloor_elevation = np.median(sediment_elevations)
        
        # -----------------------------------------------------------------
        # CALCULATE STICK-UP HEIGHTS AND VOLUME
        # -----------------------------------------------------------------
        object_elevations = elevation[object_mask]
        stick_up_heights = object_elevations - seafloor_elevation
        
        # Keep only positive heights (objects protruding upward)
        valid_heights = stick_up_heights[stick_up_heights > 0]
        num_exposed_pixels = len(valid_heights)
        
        if num_exposed_pixels == 0:
            print("[WARNING] No pixels protruding above seafloor")
            return {
                'total_exposed_volume_m3': 0.0,
                'mean_volume_per_object_m3': 0.0,
                'volume_distribution': {'counts': [], 'bin_edges': []},
                'per_object_volumes': [],
                'num_exposed_pixels': 0,
                'seafloor_elevation_m': float(seafloor_elevation)
            }
        
        # Calculate pixel area in square meters
        pixel_area_m2 = self.meters_per_pixel ** 2
        
        # Volume integration: V = Σ(height_i × pixel_area)
        volumes = valid_heights * pixel_area_m2
        total_exposed_volume = float(np.sum(volumes))
        
        # Create histogram
        hist, bin_edges = np.histogram(volumes, bins=20)
        volume_distribution = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        # -----------------------------------------------------------------
        # PER-OBJECT VOLUMES (if polygons available)
        # -----------------------------------------------------------------
        per_object_volumes = []
        if self.geojson_path is not None:
            try:
                polygons = self._load_polygons()
                print(f"  Calculating per-object volumes for {len(polygons)} objects...")
                
                for i, poly in enumerate(tqdm(polygons, desc="Per-object volumes", unit="obj")):
                    minx, miny, maxx, maxy = poly.bounds
                    
                    # Convert to pixel coordinates
                    row_min = int(max(0, miny))
                    row_max = int(min(elevation.shape[0], maxy + 1))
                    col_min = int(max(0, minx))
                    col_max = int(min(elevation.shape[1], maxx + 1))
                    
                    if row_max > row_min and col_max > col_min:
                        region_elev = elevation[row_min:row_max, col_min:col_max]
                        region_mask = mask[row_min:row_max, col_min:col_max]
                        
                        obj_pixels = region_elev[(region_mask > 0) & (~np.isnan(region_elev))]
                        if len(obj_pixels) > 0:
                            obj_heights = obj_pixels - seafloor_elevation
                            obj_heights = obj_heights[obj_heights > 0]
                            if len(obj_heights) > 0:
                                obj_volume = float(np.sum(obj_heights) * pixel_area_m2)
                                per_object_volumes.append({
                                    'polygon_id': i,
                                    'exposed_volume_m3': obj_volume
                                })
                        
            except Exception as e:
                print(f"[WARNING] Could not compute per-object volumes: {e}")
        
        # Mean volume per object
        mean_volume = (total_exposed_volume / len(per_object_volumes)) if per_object_volumes else 0.0
        
        results = {
            'total_exposed_volume_m3': total_exposed_volume,
            'mean_volume_per_object_m3': float(mean_volume),
            'volume_distribution': volume_distribution,
            'per_object_volumes': per_object_volumes,
            'num_exposed_pixels': int(num_exposed_pixels),
            'seafloor_elevation_m': float(seafloor_elevation)
        }
        
        print(f"  → Total exposed volume: {total_exposed_volume:.4f} m³")
        print(f"  → Exposed pixels: {num_exposed_pixels}")
        print(f"  → Mean volume per object: {mean_volume:.4f} m³")
        
        return results
    
    def calculate_embedment_angle(self) -> Dict[str, Any]:
        """
        Calculate Embedment Angle - The Breakout Force Metric.
        
        This metric measures the steepness of the contact slope where nodules
        meet the sediment. Steep angles indicate objects sitting on top (easy
        extraction), shallow angles indicate burial/draping (difficult extraction).
        
        Method:
            1. Extract perimeter pixels (boundary of object mask)
            2. Calculate elevation gradients (dz/dx, dz/dy)
            3. Compute slope angles at perimeter
            4. Return statistics on contact steepness
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'mean_contact_angle_deg': Mean contact slope in degrees
                - 'median_contact_angle_deg': Median contact slope
                - 'std_contact_angle_deg': Standard deviation
                - 'min_angle_deg': Minimum angle (most buried)
                - 'max_angle_deg': Maximum angle (most exposed)
                - 'perimeter_pixels_analyzed': Count of boundary pixels
                - 'per_object_angles': Per-polygon angle statistics
        
        Raises:
            ValueError: If elevation_path or mask_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(
            ...     elevation_path="depth.tif",
            ...     mask_path="nodules.tif"
            ... )
            >>> embedment = analyzer.calculate_embedment_angle()
            >>> print(f"Mean contact angle: {embedment['mean_contact_angle_deg']:.1f}°")
        """
        print("\n[METRIC] Calculating Embedment Angle...")
        
        # Load required data
        elevation = self._load_elevation()
        mask = self._load_mask()
        
        # Resample elevation to match mask dimensions if needed
        elevation = self._resample_elevation_to_mask(elevation, mask)
        
        # -----------------------------------------------------------------
        # EXTRACT PERIMETER PIXELS
        # -----------------------------------------------------------------
        from scipy.ndimage import binary_dilation
        
        dilated = binary_dilation(mask > 0, iterations=1)
        perimeter_mask = dilated & ~(mask > 0)
        
        if not np.any(perimeter_mask):
            print("[WARNING] No perimeter pixels found")
            return {
                'mean_contact_angle_deg': 0.0,
                'median_contact_angle_deg': 0.0,
                'std_contact_angle_deg': 0.0,
                'min_angle_deg': 0.0,
                'max_angle_deg': 0.0,
                'perimeter_pixels_analyzed': 0,
                'per_object_angles': []
            }
        
        # -----------------------------------------------------------------
        # COMPUTE ELEVATION GRADIENTS
        # -----------------------------------------------------------------
        dz_dx, dz_dy = np.gradient(elevation)
        
        # Extract gradients at perimeter pixels
        perimeter_indices = np.where(perimeter_mask)
        dz_dx_perim = dz_dx[perimeter_indices]
        dz_dy_perim = dz_dy[perimeter_indices]
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(dz_dx_perim**2 + dz_dy_perim**2)
        
        # -----------------------------------------------------------------
        # COMPUTE ANGLES IN DEGREES
        # -----------------------------------------------------------------
        angles_rad = np.arctan(gradient_magnitude)
        angles_deg = np.degrees(angles_rad)
        
        # Filter out NaN/Inf
        valid_angles = angles_deg[~np.isnan(angles_deg) & ~np.isinf(angles_deg)]
        
        if len(valid_angles) == 0:
            print("[WARNING] No valid angles computed")
            return {
                'mean_contact_angle_deg': 0.0,
                'median_contact_angle_deg': 0.0,
                'std_contact_angle_deg': 0.0,
                'min_angle_deg': 0.0,
                'max_angle_deg': 0.0,
                'perimeter_pixels_analyzed': len(angles_deg),
                'per_object_angles': []
            }
        
        # Calculate statistics
        mean_angle = float(np.mean(valid_angles))
        median_angle = float(np.median(valid_angles))
        std_angle = float(np.std(valid_angles))
        min_angle = float(np.min(valid_angles))
        max_angle = float(np.max(valid_angles))
        
        # -----------------------------------------------------------------
        # PER-OBJECT ANGLES (if polygons available)
        # -----------------------------------------------------------------
        per_object_angles = []
        if self.geojson_path is not None:
            try:
                polygons = self._load_polygons()
                print(f"  Calculating per-object angles for {len(polygons)} objects...")
                
                for i, poly in enumerate(tqdm(polygons, desc="Per-object angles", unit="obj")):
                    minx, miny, maxx, maxy = poly.bounds
                    
                    row_min = int(max(0, miny))
                    row_max = int(min(elevation.shape[0], maxy + 1))
                    col_min = int(max(0, minx))
                    col_max = int(min(elevation.shape[1], maxx + 1))
                    
                    if row_max > row_min and col_max > col_min:
                        region_mask = mask[row_min:row_max, col_min:col_max]
                        region_dz_dx = dz_dx[row_min:row_max, col_min:col_max]
                        region_dz_dy = dz_dy[row_min:row_max, col_min:col_max]
                        
                        # Dilate region mask
                        region_dilated = binary_dilation(region_mask > 0, iterations=1)
                        region_perim = region_dilated & ~(region_mask > 0)
                        
                        if np.any(region_perim):
                            perim_grad = np.sqrt(
                                region_dz_dx[region_perim]**2 + region_dz_dy[region_perim]**2
                            )
                            perim_angles = np.degrees(np.arctan(perim_grad))
                            perim_angles = perim_angles[~np.isnan(perim_angles) & ~np.isinf(perim_angles)]
                            
                            if len(perim_angles) > 0:
                                per_object_angles.append({
                                    'polygon_id': i,
                                    'mean_angle_deg': float(np.mean(perim_angles)),
                                    'max_angle_deg': float(np.max(perim_angles))
                                })
                        
            except Exception as e:
                print(f"[WARNING] Could not compute per-object angles: {e}")
        
        results = {
            'mean_contact_angle_deg': mean_angle,
            'median_contact_angle_deg': median_angle,
            'std_contact_angle_deg': std_angle,
            'min_angle_deg': min_angle,
            'max_angle_deg': max_angle,
            'perimeter_pixels_analyzed': int(len(valid_angles)),
            'per_object_angles': per_object_angles
        }
        
        print(f"  → Mean contact angle: {mean_angle:.2f}°")
        print(f"  → Range: [{min_angle:.2f}°, {max_angle:.2f}°]")
        print(f"  → Perimeter pixels: {len(valid_angles)}")
        
        return results
    
    def calculate_sediment_scour_anisotropy(self, ring_width_pixels: int = 5) -> Dict[str, Any]:
        """
        Calculate Sediment Scour Anisotropy - The Environmental History Metric.
        
        This metric analyzes asymmetry in sediment elevation around nodules,
        using the mud as a "current meter". An asymmetric scour (deep on one side,
        shallow on the other) indicates current direction and magnitude of disturbance.
        
        Method:
            1. Create a ring mask around objects (dilated boundary)
            2. Find minimum (scour) and maximum (wake) in ring
            3. Compute vector angle from scour to wake
            4. Calculate magnitude of anisotropy (elevation difference)
            5. Repeat per-object
        
        Args:
            ring_width_pixels: Width of ring to analyze (default: 5 pixels)
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'global': Global-scale scour results (dominant current)
                  - 'dominant_angle_deg': Direction 0-360° (0°=East, 90°=North)
                  - 'scour_depth_m': Elevation of deepest scour point
                  - 'wake_height_m': Elevation of highest wake point
                  - 'anisotropy_magnitude_m': Elevation difference
                  - 'ring_sample_size': Number of ring pixels
                - 'per_object': List of per-polygon scour results
                  - 'polygon_id': Polygon identifier
                  - 'dominant_angle_deg': Local current direction
                  - 'anisotropy_magnitude_m': Local elevation asymmetry
        
        Raises:
            ValueError: If elevation_path or mask_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(
            ...     elevation_path="depth.tif",
            ...     mask_path="nodules.tif"
            ... )
            >>> scour = analyzer.calculate_sediment_scour_anisotropy(ring_width_pixels=5)
            >>> print(f"Current direction: {scour['global']['dominant_angle_deg']:.0f}°")
        """
        print("\n[METRIC] Calculating Sediment Scour Anisotropy...")
        
        from scipy.ndimage import binary_dilation
        
        # Load required data
        elevation = self._load_elevation()
        mask = self._load_mask()
        
        if elevation.shape != mask.shape:
            raise ValueError(
                f"Elevation shape {elevation.shape} does not match mask shape {mask.shape}"
            )
        
        # -----------------------------------------------------------------
        # GLOBAL ANALYSIS: CREATE RING MASK
        # -----------------------------------------------------------------
        print("  Computing global scour anisotropy...")
        dilated = binary_dilation(mask > 0, iterations=ring_width_pixels)
        ring_mask = dilated & ~(mask > 0)
        
        global_result = {
            'dominant_angle_deg': None,
            'scour_depth_m': None,
            'wake_height_m': None,
            'anisotropy_magnitude_m': None,
            'ring_sample_size': 0
        }
        
        if np.any(ring_mask):
            # Extract ring elevations and coordinates
            ring_indices = np.where(ring_mask)
            ring_elevations = elevation[ring_indices]
            ring_x = ring_indices[1]  # Column indices
            ring_y = ring_indices[0]  # Row indices
            
            # Filter NaN values
            valid_mask = ~np.isnan(ring_elevations)
            if np.any(valid_mask):
                valid_elev = ring_elevations[valid_mask]
                valid_x = ring_x[valid_mask]
                valid_y = ring_y[valid_mask]
                
                # Find scour (minimum) and wake (maximum)
                scour_idx = np.nanargmin(ring_elevations)
                wake_idx = np.nanargmax(ring_elevations)
                
                scour_x, scour_y = ring_x[scour_idx], ring_y[scour_idx]
                wake_x, wake_y = ring_x[wake_idx], ring_y[wake_idx]
                scour_elev = ring_elevations[scour_idx]
                wake_elev = ring_elevations[wake_idx]
                
                # Compute current direction (vector from scour to wake)
                delta_x = wake_x - scour_x
                delta_y = wake_y - scour_y
                
                # Convert to meters
                delta_x_m = delta_x * self.meters_per_pixel
                delta_y_m = delta_y * self.meters_per_pixel
                
                # Angle: 0° = East, 90° = North
                angle_rad = np.arctan2(delta_y_m, delta_x_m)
                angle_deg = np.degrees(angle_rad)
                
                # Normalize to [0, 360)
                if angle_deg < 0:
                    angle_deg += 360
                
                # Anisotropy magnitude
                anisotropy_mag = wake_elev - scour_elev
                
                global_result = {
                    'dominant_angle_deg': float(angle_deg),
                    'scour_depth_m': float(scour_elev),
                    'wake_height_m': float(wake_elev),
                    'anisotropy_magnitude_m': float(anisotropy_mag),
                    'ring_sample_size': int(np.sum(valid_mask))
                }
        
        # -----------------------------------------------------------------
        # PER-OBJECT ANALYSIS (if polygons available)
        # -----------------------------------------------------------------
        per_object_results = []
        if self.geojson_path is not None:
            try:
                polygons = self._load_polygons()
                print(f"  Calculating per-object scour for {len(polygons)} objects...")
                
                for i, poly in enumerate(tqdm(polygons, desc="Per-object scour", unit="obj")):
                    minx, miny, maxx, maxy = poly.bounds
                    
                    row_min = int(max(0, miny))
                    row_max = int(min(elevation.shape[0], maxy + 1))
                    col_min = int(max(0, minx))
                    col_max = int(min(elevation.shape[1], maxx + 1))
                    
                    if row_max > row_min and col_max > col_min:
                        region_mask = mask[row_min:row_max, col_min:col_max]
                        region_elev = elevation[row_min:row_max, col_min:col_max]
                        
                        # Create ring for this object
                        region_dilated = binary_dilation(region_mask > 0, iterations=ring_width_pixels)
                        region_ring = region_dilated & ~(region_mask > 0)
                        
                        if np.any(region_ring):
                            ring_elev_region = region_elev[region_ring]
                            valid_mask_region = ~np.isnan(ring_elev_region)
                            
                            if np.any(valid_mask_region):
                                scour_elev_obj = np.nanmin(ring_elev_region)
                                wake_elev_obj = np.nanmax(ring_elev_region)
                                aniso_obj = wake_elev_obj - scour_elev_obj
                                
                                # Find indices for angle calculation
                                scour_idx_obj = np.nanargmin(ring_elev_region)
                                wake_idx_obj = np.nanargmax(ring_elev_region)
                                
                                ring_indices_obj = np.where(region_ring)
                                scour_y_obj = ring_indices_obj[0][scour_idx_obj]
                                scour_x_obj = ring_indices_obj[1][scour_idx_obj]
                                wake_y_obj = ring_indices_obj[0][wake_idx_obj]
                                wake_x_obj = ring_indices_obj[1][wake_idx_obj]
                                
                                delta_x_obj = wake_x_obj - scour_x_obj
                                delta_y_obj = wake_y_obj - scour_y_obj
                                
                                angle_obj = np.degrees(np.arctan2(delta_y_obj, delta_x_obj))
                                if angle_obj < 0:
                                    angle_obj += 360
                                
                                per_object_results.append({
                                    'polygon_id': i,
                                    'dominant_angle_deg': float(angle_obj),
                                    'anisotropy_magnitude_m': float(aniso_obj)
                                })
                        
            except Exception as e:
                print(f"[WARNING] Could not compute per-object scour: {e}")
        
        results = {
            'global': global_result,
            'per_object': per_object_results
        }
        
        if global_result['dominant_angle_deg'] is not None:
            print(f"  → Global current direction: {global_result['dominant_angle_deg']:.0f}°")
            print(f"  → Anisotropy magnitude: {global_result['anisotropy_magnitude_m']:.4f} m")
            print(f"  → Ring sample size: {global_result['ring_sample_size']} pixels")
        
        return results
    
    # =========================================================================
    # ECOSYSTEM DYNAMICS METRICS
    # =========================================================================
    
    def calculate_biodiversity_correlation(
        self,
        resource_class: str = 'nodule',
        biology_class: str = 'organism',
        grid_size: int = 5,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """
        Calculate Biodiversity-Density Correlation - The Habitat Value Metric.
        
        This metric tests whether biological abundance is correlated with resource
        (nodule) density. A strong positive correlation suggests that removing
        nodules will directly impact biodiversity, while weak/no correlation
        suggests biology exists independently of the resource.
        
        Method:
            1. Divide survey area into grid cells
            2. Count resource objects (Class A) per cell
            3. Count biological objects (Class B) per cell
            4. Calculate correlation coefficient between counts
        
        Args:
            resource_class: Class name for resource objects (e.g., 'nodule').
            biology_class: Class name for biological objects (e.g., 'organism').
            grid_size: Number of grid cells per dimension (default: 5).
            method: Correlation method - 'pearson' or 'spearman' (default: 'pearson').
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'correlation': Correlation coefficient (-1 to 1)
                - 'p_value': Statistical significance
                - 'method': Correlation method used
                - 'interpretation': Text interpretation of results
                - 'resource_counts': Resource counts per cell
                - 'biology_counts': Biology counts per cell
                - 'n_cells': Number of grid cells analyzed
        
        Raises:
            ValueError: If geojson_path was not provided.
            ValueError: If required classes not found in GeoJSON.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(geojson_path="annotations.geojson")
            >>> bio_corr = analyzer.calculate_biodiversity_correlation(
            ...     resource_class='nodule',
            ...     biology_class='sponge'
            ... )
            >>> print(f"Correlation: {bio_corr['correlation']:.3f}")
            >>> print(f"Interpretation: {bio_corr['interpretation']}")
        
        Note:
            A correlation close to 1.0 indicates "More Rocks == More Life",
            creating a conflict between mining and conservation. A correlation
            near 0 suggests biology may exist independently of nodules.
        """
        print(f"\n[METRIC] Calculating Biodiversity-Density Correlation...")
        print(f"  Resource class: '{resource_class}', Biology class: '{biology_class}'")
        
        # Load polygons with class labels
        polygons_by_class = self._load_polygons_with_classes()
        
        # Check if required classes exist
        available_classes = list(polygons_by_class.keys())
        print(f"  Available classes: {available_classes}")
        
        if resource_class not in polygons_by_class:
            print(f"[WARNING] Resource class '{resource_class}' not found. "
                  f"Available: {available_classes}")
            # Try to find a similar class
            resource_polys = []
            for cls in available_classes:
                if 'nodule' in cls.lower() or 'rock' in cls.lower():
                    resource_polys = polygons_by_class[cls]
                    print(f"  Using '{cls}' as resource class")
                    break
        else:
            resource_polys = polygons_by_class[resource_class]
        
        if biology_class not in polygons_by_class:
            print(f"[WARNING] Biology class '{biology_class}' not found. "
                  f"Available: {available_classes}")
            biology_polys = []
            for cls in available_classes:
                if any(term in cls.lower() for term in ['organism', 'sponge', 'coral', 'fauna', 'bio']):
                    biology_polys = polygons_by_class[cls]
                    print(f"  Using '{cls}' as biology class")
                    break
        else:
            biology_polys = polygons_by_class[biology_class]
        
        if len(resource_polys) == 0 or len(biology_polys) == 0:
            print("[WARNING] Insufficient data for correlation analysis")
            return {
                'correlation': 0.0,
                'p_value': 1.0,
                'method': method,
                'interpretation': 'Insufficient data - one or both classes empty',
                'resource_counts': [],
                'biology_counts': [],
                'n_cells': 0
            }
        
        # Get all centroids for grid bounds
        all_polys = resource_polys + biology_polys
        all_centroids = np.array([[p.centroid.x, p.centroid.y] for p in all_polys])
        
        min_x, min_y = all_centroids.min(axis=0)
        max_x, max_y = all_centroids.max(axis=0)
        
        # Create grid
        x_edges = np.linspace(min_x, max_x, grid_size + 1)
        y_edges = np.linspace(min_y, max_y, grid_size + 1)
        
        # Count objects per cell
        resource_centroids = np.array([[p.centroid.x, p.centroid.y] for p in resource_polys])
        biology_centroids = np.array([[p.centroid.x, p.centroid.y] for p in biology_polys])
        
        resource_counts = []
        biology_counts = []
        
        print(f"  Analyzing {grid_size}x{grid_size} grid...")
        for i in tqdm(range(grid_size), desc="Grid analysis", unit="row"):
            for j in range(grid_size):
                x_min, x_max = x_edges[j], x_edges[j + 1]
                y_min, y_max = y_edges[i], y_edges[i + 1]
                
                # Count resources in cell
                in_cell_r = (
                    (resource_centroids[:, 0] >= x_min) & (resource_centroids[:, 0] < x_max) &
                    (resource_centroids[:, 1] >= y_min) & (resource_centroids[:, 1] < y_max)
                )
                resource_counts.append(int(np.sum(in_cell_r)))
                
                # Count biology in cell
                in_cell_b = (
                    (biology_centroids[:, 0] >= x_min) & (biology_centroids[:, 0] < x_max) &
                    (biology_centroids[:, 1] >= y_min) & (biology_centroids[:, 1] < y_max)
                )
                biology_counts.append(int(np.sum(in_cell_b)))
        
        # Calculate correlation
        resource_arr = np.array(resource_counts)
        biology_arr = np.array(biology_counts)
        
        # Check for variance (correlation undefined if one variable is constant)
        if np.std(resource_arr) == 0 or np.std(biology_arr) == 0:
            print("[WARNING] One variable has no variance - correlation undefined")
            return {
                'correlation': 0.0,
                'p_value': 1.0,
                'method': method,
                'interpretation': 'Undefined - one variable has no variance',
                'resource_counts': resource_counts,
                'biology_counts': biology_counts,
                'n_cells': grid_size * grid_size
            }
        
        # Compute correlation
        if method == 'pearson':
            correlation, p_value = pearsonr(resource_arr, biology_arr)
        else:  # spearman
            correlation, p_value = spearmanr(resource_arr, biology_arr)
        
        # Interpret results
        if abs(correlation) < 0.3:
            interpretation = "Weak/no association - biology may exist independently of resource"
        elif correlation > 0.7:
            interpretation = "Strong positive - high resource density = high biodiversity (conflict zone)"
        elif correlation > 0.3:
            interpretation = "Moderate positive - some habitat association with resource"
        elif correlation < -0.7:
            interpretation = "Strong negative - biology avoids resource areas"
        else:
            interpretation = "Moderate negative - partial segregation"
        
        results = {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'method': method,
            'interpretation': interpretation,
            'resource_counts': resource_counts,
            'biology_counts': biology_counts,
            'n_cells': grid_size * grid_size
        }
        
        print(f"  → Correlation ({method}): {correlation:.4f}")
        print(f"  → P-value: {p_value:.4f}")
        print(f"  → Interpretation: {interpretation}")
        
        return results
    
    def calculate_bivariate_ripleys_k(
        self,
        class_a: str = 'nodule',
        class_b: str = 'organism',
        radii: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Bivariate Ripley's K - The Invisible Halo Metric.
        
        This metric statistically tests whether one population class clusters
        around another more than random chance would predict. It reveals
        "invisible halos" of spatial dependence between organisms and resources.
        
        Method:
            1. Extract centroids for both classes
            2. Build KDTree on class B centroids for efficient queries
            3. For each radius, count class B points within distance of class A
            4. Compare to Poisson random null model
            5. K(r) > expected indicates attraction; K(r) < expected indicates repulsion
        
        Args:
            class_a: Source class name (e.g., 'nodule'). Default: 'nodule'.
            class_b: Target class name (e.g., 'organism'). Default: 'organism'.
            radii: Optional list of radii in meters. If None, auto-generated
                  as log-spaced from 0.1m to max observed distance (~20 radii).
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'class_a': Source class name
                - 'class_b': Target class name
                - 'radii_m': List of radii analyzed (meters)
                - 'observed_counts': Observed count of class B around class A
                - 'expected_poisson_counts': Expected counts under random model
                - 'k_values': K(r) statistic (ratio of observed to expected)
                - 'interpretation': Text description of relationship
                - 'n_class_a': Number of class A objects
                - 'n_class_b': Number of class B objects
        
        Raises:
            ValueError: If geojson_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(geojson_path="annotations.geojson")
            >>> ripley = analyzer.calculate_bivariate_ripleys_k('nodule', 'sponge')
            >>> print(f"Attraction: {ripley['interpretation']}")
        
        Note:
            A peak in K(r) above the Poisson line indicates clustering of class B
            around class A at that particular scale. This reveals which organisms
            are spatially dependent on the resource.
        """
        print(f"\n[METRIC] Calculating Bivariate Ripley's K...")
        print(f"  Class A: '{class_a}', Class B: '{class_b}'")
        
        # Load polygons with class labels
        polygons_by_class = self._load_polygons_with_classes()
        available_classes = list(polygons_by_class.keys())
        print(f"  Available classes: {available_classes}")
        
        # Mirror existing error handling pattern
        if class_a not in polygons_by_class:
            print(f"[WARNING] Class A '{class_a}' not found. Available: {available_classes}")
            class_a_polys = []
            for cls in available_classes:
                if 'nodule' in cls.lower() or 'rock' in cls.lower():
                    class_a_polys = polygons_by_class[cls]
                    print(f"  Using '{cls}' as class A")
                    break
        else:
            class_a_polys = polygons_by_class[class_a]
        
        if class_b not in polygons_by_class:
            print(f"[WARNING] Class B '{class_b}' not found. Available: {available_classes}")
            class_b_polys = []
            for cls in available_classes:
                if any(term in cls.lower() for term in ['organism', 'sponge', 'coral', 'fauna', 'bio']):
                    class_b_polys = polygons_by_class[cls]
                    print(f"  Using '{cls}' as class B")
                    break
        else:
            class_b_polys = polygons_by_class[class_b]
        
        if len(class_a_polys) < 1 or len(class_b_polys) < 1:
            print("[WARNING] Insufficient data for Ripley's K analysis")
            return {
                'class_a': class_a,
                'class_b': class_b,
                'radii_m': [],
                'observed_counts': [],
                'expected_poisson_counts': [],
                'k_values': [],
                'interpretation': 'Insufficient data - one or both classes empty',
                'n_class_a': len(class_a_polys),
                'n_class_b': len(class_b_polys)
            }
        
        # Extract centroids
        print(f"  Extracting centroids: {len(class_a_polys)} class A, {len(class_b_polys)} class B")
        centroids_a = np.array([[p.centroid.x, p.centroid.y] for p in class_a_polys])
        centroids_b = np.array([[p.centroid.x, p.centroid.y] for p in class_b_polys])
        
        # Convert to meters
        centroids_a_m = centroids_a * self.meters_per_pixel
        centroids_b_m = centroids_b * self.meters_per_pixel
        
        # -----------------------------------------------------------------
        # GENERATE RADII IF NOT PROVIDED
        # -----------------------------------------------------------------
        if radii is None:
            max_distance = np.max(np.abs(centroids_b_m)) + np.max(np.abs(centroids_a_m))
            if max_distance > 0:
                min_radius = 0.1  # meters
                radii = np.logspace(np.log10(min_radius), np.log10(max_distance), num=20)
            else:
                radii = np.logspace(-1, 1, num=20)
        
        radii = np.array(radii)
        print(f"  Analyzing {len(radii)} radii: {radii[0]:.2f}m to {radii[-1]:.2f}m")
        
        # -----------------------------------------------------------------
        # BUILD KDTREE AND COUNT NEIGHBORS
        # -----------------------------------------------------------------
        tree = KDTree(centroids_b_m)
        
        observed_counts = []
        for r in tqdm(radii, desc="Ripley's K analysis", unit="radius"):
            count = 0
            for pt_a in centroids_a_m:
                neighbors = tree.query_ball_point(pt_a, r)
                count += len(neighbors)
            observed_counts.append(int(count))
        
        # -----------------------------------------------------------------
        # COMPUTE EXPECTED POISSON DISTRIBUTION
        # -----------------------------------------------------------------
        # Estimate total area from mask or bounding box
        if self.mask_path is not None:
            mask = self._load_mask()
            total_area_m2 = mask.size * (self.meters_per_pixel ** 2)
        else:
            # Estimate from bounding box
            all_centroids = np.vstack([centroids_a_m, centroids_b_m])
            x_range = np.max(all_centroids[:, 0]) - np.min(all_centroids[:, 0])
            y_range = np.max(all_centroids[:, 1]) - np.min(all_centroids[:, 1])
            total_area_m2 = x_range * y_range
        
        # Density of class B
        density_b = len(centroids_b) / total_area_m2 if total_area_m2 > 0 else 0
        
        # Expected counts under Poisson
        expected_counts = [density_b * np.pi * r**2 * len(centroids_a) for r in radii]
        
        # -----------------------------------------------------------------
        # COMPUTE K-VALUES
        # -----------------------------------------------------------------
        density_a = len(centroids_a) / total_area_m2 if total_area_m2 > 0 else 1.0
        k_values = []
        for obs, r in zip(observed_counts, radii):
            if r > 0:
                k_val = obs / (density_a * np.pi * r**2) if density_a > 0 else 0
            else:
                k_val = 0
            k_values.append(float(k_val))
        
        # -----------------------------------------------------------------
        # INTERPRETATION
        # -----------------------------------------------------------------
        if len(k_values) > 0:
            mean_expected = np.mean(expected_counts)
            mean_k = np.mean(k_values)
            
            if mean_k > mean_expected * 1.2:
                interpretation = "Attraction: class B clusters around class A"
            elif mean_k < mean_expected * 0.8:
                interpretation = "Repulsion: class B avoids class A"
            else:
                interpretation = "Random distribution: no significant spatial dependence"
        else:
            interpretation = "Unable to interpret"
        
        results = {
            'class_a': class_a,
            'class_b': class_b,
            'radii_m': radii.tolist(),
            'observed_counts': observed_counts,
            'expected_poisson_counts': [float(e) for e in expected_counts],
            'k_values': k_values,
            'interpretation': interpretation,
            'n_class_a': len(class_a_polys),
            'n_class_b': len(class_b_polys)
        }
        
        print(f"  → Mean K-value: {mean_k:.4f}")
        print(f"  → Interpretation: {interpretation}")
        
        return results
    
    def calculate_beta_diversity_turnover(
        self,
        grid_size: int = 4,
        diversity_metric: str = 'jaccard'
    ) -> Dict[str, Any]:
        """
        Calculate Beta Diversity Turnover - The Community Drift Metric.
        
        This metric quantifies how species composition changes across the survey
        site (community drift). It measures whether the same organisms appear
        everywhere or if communities shift in space.
        
        Method:
            1. Divide survey area into grid cells
            2. Identify which biological classes are present in each cell
            3. Compare adjacent cells using Jaccard or Sorensen distance
            4. Average distances across all cell pairs
        
        Args:
            grid_size: Number of grid cells per dimension (default: 4).
                      Total cells = grid_size × grid_size.
            diversity_metric: Distance metric - 'jaccard' (default) or 'sorensen'.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'diversity_metric': Metric used ('jaccard' or 'sorensen')
                - 'mean_distance': Mean pairwise distance (0-1)
                - 'std_distance': Standard deviation of distances
                - 'min_distance': Minimum pairwise distance
                - 'max_distance': Maximum pairwise distance
                - 'grid_size': Grid dimensions used
                - 'n_cells_analyzed': Number of cells
                - 'n_cells_with_biology': Number of cells with organisms
                - 'per_cell_species': Species list per grid cell (for detailed analysis)
                - 'interpretation': Text summary
        
        Raises:
            ValueError: If geojson_path was not provided.
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(geojson_path="annotations.geojson")
            >>> beta = analyzer.calculate_beta_diversity_turnover(grid_size=5)
            >>> print(f"Community drift: {beta['interpretation']}")
        
        Note:
            Values close to 0 indicate stable communities (low turnover).
            Values close to 1 indicate high species turnover (community drift).
        """
        print(f"\n[METRIC] Calculating Beta Diversity Turnover (metric={diversity_metric})...")
        
        if diversity_metric not in ['jaccard', 'sorensen']:
            raise ValueError("diversity_metric must be 'jaccard' or 'sorensen'")
        
        # Load polygons with class labels
        polygons_by_class = self._load_polygons_with_classes()
        
        if not polygons_by_class:
            print("[WARNING] No polygons found")
            return {
                'diversity_metric': diversity_metric,
                'mean_distance': 0.0,
                'std_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'grid_size': grid_size,
                'n_cells_analyzed': grid_size * grid_size,
                'n_cells_with_biology': 0,
                'per_cell_species': {},
                'interpretation': 'No biological data'
            }
        
        # Flatten polygons and get bounding box
        all_polygons = sum(polygons_by_class.values(), [])
        all_centroids = np.array([[p.centroid.x, p.centroid.y] for p in all_polygons])
        
        min_x, min_y = all_centroids.min(axis=0)
        max_x, max_y = all_centroids.max(axis=0)
        
        # Create grid
        print(f"  Creating {grid_size}x{grid_size} grid")
        x_edges = np.linspace(min_x, max_x, grid_size + 1)
        y_edges = np.linspace(min_y, max_y, grid_size + 1)
        
        # Identify species per cell
        grid_species = {}  # key: (i, j), value: set of class names
        
        print(f"  Identifying species in each grid cell...")
        for i in tqdm(range(grid_size), desc="Grid analysis", unit="row"):
            for j in range(grid_size):
                x_min, x_max = x_edges[j], x_edges[j + 1]
                y_min, y_max = y_edges[i], y_edges[i + 1]
                
                cell_species = set()
                for class_name, polys in polygons_by_class.items():
                    for poly in polys:
                        if x_min <= poly.centroid.x < x_max and y_min <= poly.centroid.y < y_max:
                            cell_species.add(class_name)
                            break  # Only need 1 representative per class per cell
                
                grid_species[(i, j)] = cell_species
        
        # -----------------------------------------------------------------
        # DEFINE DISTANCE METRICS
        # -----------------------------------------------------------------
        def jaccard_distance(set_a, set_b):
            union_size = len(set_a | set_b)
            if union_size == 0:
                return 0.0
            return 1.0 - (len(set_a & set_b) / union_size)
        
        def sorensen_distance(set_a, set_b):
            denom = len(set_a) + len(set_b)
            if denom == 0:
                return 0.0
            return 1.0 - (2 * len(set_a & set_b) / denom)
        
        distance_func = jaccard_distance if diversity_metric == 'jaccard' else sorensen_distance
        
        # -----------------------------------------------------------------
        # COMPUTE PAIRWISE DISTANCES
        # -----------------------------------------------------------------
        all_cells = list(grid_species.keys())
        distances = []
        
        print(f"  Computing pairwise distances...")
        for i, cell_a in enumerate(all_cells):
            for cell_b in all_cells[i + 1:]:
                d = distance_func(grid_species[cell_a], grid_species[cell_b])
                distances.append(d)
        
        # Calculate statistics
        if len(distances) > 0:
            distances_arr = np.array(distances)
            mean_distance = float(np.mean(distances_arr))
            std_distance = float(np.std(distances_arr))
            min_distance = float(np.min(distances_arr))
            max_distance = float(np.max(distances_arr))
        else:
            mean_distance = std_distance = min_distance = max_distance = 0.0
        
        # Count cells with biology
        n_cells_with_biology = sum(1 for species_set in grid_species.values() if len(species_set) > 0)
        
        # -----------------------------------------------------------------
        # INTERPRETATION
        # -----------------------------------------------------------------
        if mean_distance < 0.2:
            interpretation = "Community stable: high species consistency across site"
        elif mean_distance < 0.5:
            interpretation = "Moderate turnover: gradual species composition change"
        else:
            interpretation = "High turnover: strong community drift across site"
        
        # Prepare per-cell species dict for output
        per_cell_species_dict = {
            f"({i},{j})": list(species_set)
            for (i, j), species_set in grid_species.items()
        }
        
        results = {
            'diversity_metric': diversity_metric,
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'grid_size': grid_size,
            'n_cells_analyzed': grid_size * grid_size,
            'n_cells_with_biology': n_cells_with_biology,
            'per_cell_species': per_cell_species_dict,
            'interpretation': interpretation
        }
        
        print(f"  → Mean distance ({diversity_metric}): {mean_distance:.4f}")
        print(f"  → Cells with biology: {n_cells_with_biology}/{grid_size * grid_size}")
        print(f"  → Interpretation: {interpretation}")
        
        return results
    
    def calculate_projected_biological_loss(
        self,
        mining_polygon: Polygon,
        buffer_distance_m: float = 5.0
    ) -> Dict[str, Any]:
        """
        Calculate Projected Biological Loss - The Impact Simulation Metric.
        
        This metric simulates the ecological impact of a mining operation by
        identifying which biological organisms would be directly killed (within
        mining footprint) or indirectly affected (within buffer zone).
        
        Method:
            1. Use mining_polygon.within() to find organisms in direct footprint
            2. Create buffer zone around mining polygon
            3. Use intersects() to find organisms in indirect impact zone
            4. Sum casualties by class
        
        Args:
            mining_polygon: Shapely Polygon representing the mining operation footprint.
            buffer_distance_m: Buffer zone distance in meters (default: 5.0).
                              Organisms in buffer treated as indirectly affected.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'mining_polygon_area_m2': Area of mining footprint
                - 'buffer_distance_m': Buffer distance applied
                - 'direct_casualties': Organisms directly killed
                  - 'count': Total count
                  - 'area_m2': Total area (if applicable)
                  - 'by_class': {'class_name': {'count': int, 'area_m2': float}, ...}
                - 'indirect_casualties': Organisms indirectly affected
                  - 'count': Total count
                  - 'area_m2': Total area
                  - 'by_class': {...}
                - 'total_impact': Summary of all casualties
                  - 'count': Total organisms impacted
                  - 'area_m2': Total area impacted
        
        Raises:
            ValueError: If geojson_path was not provided.
            ValueError: If mining_polygon is invalid or empty.
        
        Example:
            >>> from shapely.geometry import Polygon
            >>> analyzer = SpatialMetricsAnalyzer(geojson_path="fauna.geojson")
            >>> mining_zone = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
            >>> impact = analyzer.calculate_projected_biological_loss(
            ...     mining_polygon=mining_zone,
            ...     buffer_distance_m=10
            ... )
            >>> print(f"Direct casualties: {impact['direct_casualties']['count']}")
        
        Note:
            This is a **deterministic simulation** using strict geometric criteria.
            Real-world impacts may be more complex (e.g., partial mortality,
            recovery rates). Use as a baseline worst-case scenario.
        """
        print(f"\n[METRIC] Calculating Projected Biological Loss...")
        print(f"  Mining footprint area: {mining_polygon.area * (self.meters_per_pixel ** 2):.2f} m²")
        print(f"  Buffer distance: {buffer_distance_m} m")
        
        # Validate mining polygon
        if mining_polygon is None or mining_polygon.is_empty:
            print("[WARNING] Invalid or empty mining polygon")
            return {
                'mining_polygon_area_m2': 0.0,
                'buffer_distance_m': buffer_distance_m,
                'direct_casualties': {
                    'count': 0,
                    'area_m2': 0.0,
                    'by_class': {}
                },
                'indirect_casualties': {
                    'count': 0,
                    'area_m2': 0.0,
                    'by_class': {}
                },
                'total_impact': {
                    'count': 0,
                    'area_m2': 0.0
                }
            }
        
        # Load polygons with class labels
        polygons_by_class = self._load_polygons_with_classes()
        
        if not polygons_by_class:
            print("[WARNING] No biological polygons found")
            return {
                'mining_polygon_area_m2': mining_polygon.area * (self.meters_per_pixel ** 2),
                'buffer_distance_m': buffer_distance_m,
                'direct_casualties': {
                    'count': 0,
                    'area_m2': 0.0,
                    'by_class': {}
                },
                'indirect_casualties': {
                    'count': 0,
                    'area_m2': 0.0,
                    'by_class': {}
                },
                'total_impact': {
                    'count': 0,
                    'area_m2': 0.0
                }
            }
        
        # -----------------------------------------------------------------
        # CREATE HAZARD BUFFER
        # -----------------------------------------------------------------
        buffer_pixels = buffer_distance_m / self.meters_per_pixel
        hazard_zone = mining_polygon.buffer(buffer_pixels)
        
        # -----------------------------------------------------------------
        # FIND DIRECT AND INDIRECT CASUALTIES
        # -----------------------------------------------------------------
        direct_casualties = {
            'count': 0,
            'area_m2': 0.0,
            'by_class': {}
        }
        
        indirect_casualties = {
            'count': 0,
            'area_m2': 0.0,
            'by_class': {}
        }
        
        pixel_area_m2 = self.meters_per_pixel ** 2
        
        print(f"  Analyzing casualties...")
        for class_name, polys in tqdm(polygons_by_class.items(), desc="Class analysis", unit="class"):
            direct_casualties['by_class'][class_name] = {'count': 0, 'area_m2': 0.0}
            indirect_casualties['by_class'][class_name] = {'count': 0, 'area_m2': 0.0}
            
            for poly in polys:
                poly_area_m2 = poly.area * pixel_area_m2
                
                # Direct casualties (within mining footprint)
                if poly.within(mining_polygon):
                    direct_casualties['count'] += 1
                    direct_casualties['area_m2'] += poly_area_m2
                    direct_casualties['by_class'][class_name]['count'] += 1
                    direct_casualties['by_class'][class_name]['area_m2'] += poly_area_m2
                
                # Indirect casualties (within buffer but not in footprint)
                elif poly.intersects(hazard_zone):
                    indirect_casualties['count'] += 1
                    indirect_casualties['area_m2'] += poly_area_m2
                    indirect_casualties['by_class'][class_name]['count'] += 1
                    indirect_casualties['by_class'][class_name]['area_m2'] += poly_area_m2
        
        # -----------------------------------------------------------------
        # COMPUTE TOTALS
        # -----------------------------------------------------------------
        total_impact = {
            'count': direct_casualties['count'] + indirect_casualties['count'],
            'area_m2': direct_casualties['area_m2'] + indirect_casualties['area_m2']
        }
        
        results = {
            'mining_polygon_area_m2': mining_polygon.area * pixel_area_m2,
            'buffer_distance_m': buffer_distance_m,
            'direct_casualties': direct_casualties,
            'indirect_casualties': indirect_casualties,
            'total_impact': total_impact
        }
        
        print(f"  → Direct casualties: {direct_casualties['count']} organisms, "
              f"{direct_casualties['area_m2']:.2f} m²")
        print(f"  → Indirect casualties: {indirect_casualties['count']} organisms, "
              f"{indirect_casualties['area_m2']:.2f} m²")
        print(f"  → Total impact: {total_impact['count']} organisms, "
              f"{total_impact['area_m2']:.2f} m²")
        
        return results
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    def generate_report(
        self,
        output_path: Optional[str] = None,
        include_morphology_details: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate a comprehensive spatial metrics report.
        
        This method automatically detects which data inputs are available and
        runs all feasible metrics. Results are returned as both a pandas DataFrame
        (for tabular analysis) and a dictionary (for JSON serialization).
        
        The report includes:
            - Summary statistics for all computed metrics
            - Per-object morphology data (if polygons available)
            - Metadata about inputs and scale
        
        Args:
            output_path: Optional path to save JSON report. If None, no file is saved.
            include_morphology_details: Whether to include per-polygon morphology
                                       stats in the output (default: True).
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]:
                - DataFrame with summary metrics (one row per metric category)
                - Dictionary with full results (suitable for JSON export)
        
        Example:
            >>> analyzer = SpatialMetricsAnalyzer(
            ...     mask_path="mask.tif",
            ...     geojson_path="annotations.geojson"
            ... )
            >>> df, report = analyzer.generate_report(output_path="results.json")
            >>> print(df)  # Summary table
            >>> print(report['metrics']['pcf']['pcf_percent'])  # Access specific value
        
        Note:
            Metrics requiring unavailable data are skipped with a warning.
            The report includes metadata about which metrics were computed
            and which were skipped due to missing inputs.
        """
        print("\n" + "=" * 60)
        print("GENERATING SPATIAL METRICS REPORT")
        print("=" * 60)
        
        # Initialize results storage
        report: Dict[str, Any] = {
            'metadata': {
                'image_path': str(self.image_path) if self.image_path else None,
                'elevation_path': str(self.elevation_path) if self.elevation_path else None,
                'mask_path': str(self.mask_path) if self.mask_path else None,
                'geojson_path': str(self.geojson_path) if self.geojson_path else None,
                'meters_per_pixel': self.meters_per_pixel,
                'scale_factor': self.scale_factor
            },
            'metrics': {},
            'morphology_details': None,
            'computed_metrics': [],
            'skipped_metrics': []
        }
        
        summary_rows = []
        
        # Define metrics and their requirements
        metrics_config = [
            # (name, method, requires_mask, requires_geojson, requires_elevation)
            ('pcf', self.calculate_pcf, True, False, False),
            ('abundance', self.calculate_abundance, False, True, False),
            ('spatial_homogeneity', lambda: self.calculate_spatial_homogeneity(grid_size=4), 
             False, True, False),
            ('nearest_neighbor_distance', lambda: self.calculate_nearest_neighbor_distance(method='edge'),
             False, True, False),
            ('passability', self.calculate_passability_index, True, False, False),
            ('ripleys_k', lambda: self.calculate_ripleys_k(n_radii=15), False, True, False),
            ('clark_evans', self.calculate_clark_evans, False, True, False),
            ('protrusion', self.calculate_protrusion, True, False, True),
            ('rugosity_3d', self.calculate_3d_rugosity, True, False, True),
            ('exposed_volume', self.calculate_exposed_volume, True, False, True),
            ('embedment_angle', self.calculate_embedment_angle, True, False, True),
            ('sediment_scour_anisotropy', lambda: self.calculate_sediment_scour_anisotropy(ring_width_pixels=5),
             True, False, True),
            ('bivariate_ripleys_k', lambda: self.calculate_bivariate_ripleys_k('nodule', 'organism'),
             False, True, False),
            ('beta_diversity_turnover', lambda: self.calculate_beta_diversity_turnover(grid_size=4),
             False, True, False),
        ]
        
        # Run feasible metrics
        total_metrics = len(metrics_config)
        print(f"\nAnalyzing {total_metrics} metric categories...\n")
        
        for name, method, req_mask, req_geojson, req_elevation in tqdm(
            metrics_config, desc="Computing metrics", unit="metric"
        ):
            # Check requirements
            can_compute = True
            missing = []
            
            if req_mask and self.mask_path is None:
                can_compute = False
                missing.append('mask')
            if req_geojson and self.geojson_path is None:
                can_compute = False
                missing.append('geojson')
            if req_elevation and self.elevation_path is None:
                can_compute = False
                missing.append('elevation')
            
            if not can_compute:
                print(f"[SKIP] {name}: requires {', '.join(missing)}")
                report['skipped_metrics'].append({
                    'name': name,
                    'reason': f"Missing: {', '.join(missing)}"
                })
                continue
            
            # Compute metric
            try:
                result = method()
                report['metrics'][name] = result
                report['computed_metrics'].append(name)
                
                # Extract key value for summary
                if isinstance(result, dict):
                    # Get first numeric value for summary
                    key_value = None
                    key_name = None
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            key_value = v
                            key_name = k
                            break
                    
                    if key_value is not None:
                        summary_rows.append({
                            'metric': name,
                            'key_measure': key_name,
                            'value': key_value
                        })
                        
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                report['skipped_metrics'].append({
                    'name': name,
                    'reason': str(e)
                })
        
        # Compute morphology stats if polygons available
        if self.geojson_path is not None:
            try:
                morphology_df = self.calculate_morphology_stats()
                
                # Add summary stats
                if len(morphology_df) > 0:
                    morph_summary = {
                        'n_polygons': len(morphology_df),
                        'circularity_mean': float(morphology_df['circularity'].mean()),
                        'circularity_std': float(morphology_df['circularity'].std()),
                        'solidity_mean': float(morphology_df['solidity'].mean()),
                        'solidity_std': float(morphology_df['solidity'].std()),
                        'obb_aspect_ratio_mean': float(morphology_df['obb_aspect_ratio'].mean()),
                        'obb_aspect_ratio_std': float(morphology_df['obb_aspect_ratio'].std()),
                        'area_m2_mean': float(morphology_df['area_m2'].mean()),
                        'area_m2_total': float(morphology_df['area_m2'].sum())
                    }
                    report['metrics']['morphology_summary'] = morph_summary
                    report['computed_metrics'].append('morphology')
                    
                    summary_rows.append({
                        'metric': 'morphology',
                        'key_measure': 'circularity_mean',
                        'value': morph_summary['circularity_mean']
                    })
                    
                    # Include full details if requested
                    if include_morphology_details:
                        report['morphology_details'] = morphology_df.to_dict(orient='records')
                        
            except Exception as e:
                print(f"[ERROR] morphology: {e}")
                report['skipped_metrics'].append({
                    'name': 'morphology',
                    'reason': str(e)
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        
        # Print summary table
        print("\n" + "=" * 60)
        print("REPORT SUMMARY")
        print("=" * 60)
        print(f"Computed: {len(report['computed_metrics'])} metrics")
        print(f"Skipped: {len(report['skipped_metrics'])} metrics")
        print("\nKey Results:")
        print("-" * 40)
        if len(summary_df) > 0:
            for _, row in summary_df.iterrows():
                print(f"  {row['metric']:25s} {row['key_measure']:20s} = {row['value']:.4f}")
        print("=" * 60)
        
        # Save to JSON if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            print(f"\nSaving report to: {output_path}")
            
            # Convert any numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj
            
            json_report = convert_numpy(report)
            
            with open(output_path, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            print(f"Report saved successfully!")
            
            # Also save summary CSV
            csv_path = output_path.with_suffix('.csv')
            summary_df.to_csv(csv_path, index=False)
            print(f"Summary CSV saved to: {csv_path}")
        
        return summary_df, report


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTION
# =============================================================================

def analyze(
    mask_path: Optional[str] = None,
    geojson_path: Optional[str] = None,
    elevation_path: Optional[str] = None,
    image_path: Optional[str] = None,
    scale_factor: float = 1.0,
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to run full spatial metrics analysis.
    
    This is a shortcut for creating a SpatialMetricsAnalyzer and generating
    a report in a single function call.
    
    Args:
        mask_path: Path to binary segmentation mask raster.
        geojson_path: Path to GeoJSON polygon annotations.
        elevation_path: Path to elevation/depth raster (optional).
        image_path: Path to optical/acoustic image (optional).
        scale_factor: Meters per pixel if CRS is not projected.
        output_path: Path to save JSON report (optional).
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Summary DataFrame and full report dict.
    
    Example:
        >>> from spatial_metrics import analyze
        >>> df, report = analyze(
        ...     mask_path="nodules.tif",
        ...     geojson_path="annotations.geojson",
        ...     output_path="analysis.json"
        ... )
    """
    analyzer = SpatialMetricsAnalyzer(
        image_path=image_path,
        elevation_path=elevation_path,
        mask_path=mask_path,
        geojson_path=geojson_path,
        scale_factor=scale_factor
    )
    
    return analyzer.generate_report(output_path=output_path)
