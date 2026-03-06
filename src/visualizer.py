"""
SpatialMetricsVisualizer: Comprehensive visualization engine for spatial metrics.

This module provides the SpatialMetricsVisualizer class for generating publication-ready
visualizations of spatial metrics computed by SpatialMetricsAnalyzer.

All outputs assume orthorectified (nadir/top-down) imagery with perspective distortion removed.
If oblique/perspective imagery is used, visual metrics will be skewed by perspective effects.

Author: Jordan Pierce
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb
import rasterio
from shapely.geometry import Polygon
from scipy import ndimage
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import warnings

# Note: mpl_toolkits.mplot3d.Axes3D is implicitly imported when using projection='3d'
# Modern matplotlib (>=3.2) handles this automatically

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CLASS DEFINITION
# =============================================================================


class SpatialMetricsVisualizer:
    """
    Comprehensive visualization engine for spatial metrics.
    
    This class generates high-quality, publication-ready visualizations for metrics
    computed by SpatialMetricsAnalyzer. For each metric, it produces a "visualization suite":
    - Standalone clean figure: metric on white background
    - Standalone overlay figure: metric overlaid on orthomosaic imagery
    - Combined 1x2 subplot: clean and overlay side-by-side comparison
    
    All visualizations assume orthorectified (nadir/top-down) imagery where perspective
    distortion has been removed. Oblique imagery will produce visually skewed results.
    
    Attributes:
        analyzer: SpatialMetricsAnalyzer instance (provides metric data and scaling)
        output_dir (Path): Directory where figures are saved
        orthomosaic_array (np.ndarray): Cached RGB/grayscale orthomosaic image (H×W or H×W×3)
        affine_transform: Rasterio affine transform for pixel-to-world coordinate conversion
        figure_dpi (int): Resolution of saved figures in dots per inch
        figure_size (Tuple[int, int]): Figure dimensions in inches (width, height)
        _exemplar_indices (List[int]): Indices of selected exemplar polygons (for solidity viz)
    
    Example:
        >>> analyzer = SpatialMetricsAnalyzer(
        ...     image_path="orthomosaic.tif",
        ...     mask_path="mask.tif",
        ...     geojson_path="annotations.geojson"
        ... )
        >>> viz = SpatialMetricsVisualizer(analyzer, output_dir="figures/")
        >>> paths = viz.visualize_nearest_neighbor_distance()
        >>> print(f"Clean figure: {paths['clean']}")
    """
    
    def __init__(
        self,
        analyzer, 
        output_dir: str = "figures",
        figure_dpi: int = 300,
        figure_size: Tuple[int, int] = (12, 10)
    ):
        """
        Initialize the SpatialMetricsVisualizer.
        
        Args:
            analyzer: SpatialMetricsAnalyzer instance with loaded data and computed metrics
            output_dir: Directory to save generated figures (automatically created if missing)
            figure_dpi: Resolution in dots per inch for PNG output (default: 300 for publication)
            figure_size: Figure dimensions in inches as (width, height) for 1x2 combined plots
        
        Raises:
            ValueError: If analyzer is not a valid SpatialMetricsAnalyzer instance
        """
        if not hasattr(analyzer, '_load_polygons') or not hasattr(analyzer, 'meters_per_pixel'):
            raise ValueError("analyzer must be a SpatialMetricsAnalyzer instance")
        
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figure_dpi = figure_dpi
        self.figure_size = figure_size
        
        self.orthomosaic_array: Optional[np.ndarray] = None
        self.elevation_array: Optional[np.ndarray] = None
        self.affine_transform: Optional[Any] = None
        self.elevation_transform: Optional[Any] = None
        self._cached_polygons: Optional[List[Polygon]] = None
        self._exemplar_indices: Optional[List[int]] = None
        
        self._load_orthomosaic()
        self._load_elevation()
    
    # =========================================================================
    # CORE INFRASTRUCTURE METHODS
    # =========================================================================
    
    def _load_orthomosaic(self) -> None:
        """
        Load and cache the orthorectified orthomosaic image from disk.
        
        This method reads the RGB/grayscale image from analyzer.image_path using rasterio,
        handles various band configurations (single-band, 3-band RGB, 4-band RGBA),
        normalizes pixel values to 0-1 range for matplotlib display, and extracts the
        affine geospatial transform for coordinate conversion.
        
        If image loading fails or analyzer.image_path is None, orthomosaic_array is set
        to None and overlay plots will fallback to white background.
        
        Side Effects:
            Sets self.orthomosaic_array and self.affine_transform
            Prints warning if loading fails
        """
        if self.analyzer.image_path is None:
            self.orthomosaic_array = None
            return
        
        try:
            with rasterio.open(self.analyzer.image_path) as src:
                self.orthomosaic_array = src.read()
                self.affine_transform = src.transform
                
                if self.orthomosaic_array.shape[0] == 1:
                    self.orthomosaic_array = self.orthomosaic_array[0]
                elif self.orthomosaic_array.shape[0] in [3, 4]:
                    self.orthomosaic_array = np.transpose(self.orthomosaic_array, (1, 2, 0))
                    if self.orthomosaic_array.shape[2] == 4:
                        self.orthomosaic_array = self.orthomosaic_array[:, :, :3]
                
                if self.orthomosaic_array.dtype == np.uint8:
                    self.orthomosaic_array = self.orthomosaic_array.astype(np.float32) / 255.0
                elif self.orthomosaic_array.dtype == np.uint16:
                    self.orthomosaic_array = self.orthomosaic_array.astype(np.float32) / 65535.0
                else:
                    self.orthomosaic_array = np.clip(self.orthomosaic_array, 0, 1)
        except Exception as e:
            print(f"[WARNING] Failed to load orthomosaic: {e}")
            self.orthomosaic_array = None
    
    def _load_elevation(self) -> None:
        """
        Load and cache the elevation/depth raster.
        
        CRITICAL ASSUMPTION: Elevation values are in METERS, matching the XY coordinate
        system scale. If elevation is in millimeters or raw acoustic units, Z-axis
        will be severely distorted relative to X/Y.
        
        This method reads the elevation raster from analyzer.elevation_path, handles
        nodata values by converting to NaN. The elevation raster may have different
        pixel dimensions than the orthomosaic as long as they cover the same geographic
        area (co-registration is handled via affine transforms during rendering).
        
        Side Effects:
            Sets self.elevation_array and self.elevation_transform
            Prints warning if loading fails or elevation_path is None
        """
        if self.analyzer.elevation_path is None:
            print("[INFO] No elevation path provided; 3D plots will be skipped.")
            self.elevation_array = None
            self.elevation_transform = None
            return
        
        try:
            print("[INFO] Loading elevation raster...")
            with rasterio.open(self.analyzer.elevation_path) as src:
                self.elevation_array = src.read(1).astype(np.float32)
                self.elevation_transform = src.transform
                
                # Handle nodata values by replacing with NaNs
                if src.nodata is not None:
                    self.elevation_array[self.elevation_array == src.nodata] = np.nan
            
            # If orthomosaic is available, ensure DEM matches its pixel grid.
            if self.orthomosaic_array is not None and self.elevation_array is not None:
                ortho_shape = self.orthomosaic_array.shape[:2]
                elev_shape = self.elevation_array.shape
                if elev_shape != ortho_shape:
                    print(f"[WARNING] Elevation resolution {elev_shape} differs from "
                          f"orthomosaic {ortho_shape}. Resampling DEM to orthomosaic pixel grid.")
                    try:
                        # Use analyzer helper to resample elevation to target shape
                        self.elevation_array = self.analyzer._resample_elevation_to_shape(
                            self.elevation_array, ortho_shape
                        )

                        # After resampling, adopt orthomosaic affine if available so
                        # that subsequent world<->pixel conversions align to the
                        # orthomosaic pixel grid used for plotting.
                        if self.affine_transform is not None:
                            self.elevation_transform = self.affine_transform
                        else:
                            # Fall back to original elevation transform if present
                            self.elevation_transform = self.elevation_transform

                        print(f"[INFO] Elevation resampled: new shape={self.elevation_array.shape}")
                    except Exception as e:
                        print(f"[WARNING] Resampling elevation failed: {e}. Continuing with original DEM shape.")
            
            print(f"[INFO] Elevation loaded: shape={self.elevation_array.shape}")
            print("[INFO] ASSUMPTION: Elevation units are METERS (matching XY scale)")
            
        except Exception as e:
            print(f"[WARNING] Failed to load elevation: {e}")
            self.elevation_array = None
            self.elevation_transform = None
            
    def _world_to_pixel(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Convert world coordinates (meters) to pixel coordinates (image space).
        
        Uses the affine geospatial transform to convert from projected coordinate system
        (e.g., UTM with units in meters) to pixel indices in the orthomosaic image.
        
        Args:
            world_x: X coordinate in meters (easting)
            world_y: Y coordinate in meters (northing)
        
        Returns:
            Tuple[float, float]: (pixel_row, pixel_col) in image space
        
        Note:
            If affine transform is not available, assumes 1:1 meter-to-pixel mapping.
        """
        if self.affine_transform is not None:
            inv_transform = ~self.affine_transform
            pixel_col, pixel_row = inv_transform * (world_x, world_y)
            return pixel_row, pixel_col
        else:
            return float(world_y) / self.analyzer.meters_per_pixel, float(world_x) / self.analyzer.meters_per_pixel
    
    def _get_axis_limits(self) -> Tuple[float, float, float, float]:
        """
        Calculate axis limits in world coordinates that encompass the entire region of interest.
        
        Determines the geographic extent of the study area by reading raster dimensions and
        converting from pixel space to world coordinates (meters).
        
        Returns:
            Tuple[float, float, float, float]: (x_min, x_max, y_min, y_max) in meters
                representing the bounding box of the ROI
        
        Note:
            Falls back to default [0, 100] range if no raster data is available.
        """
        if self.orthomosaic_array is not None:
            height, width = self.orthomosaic_array.shape[:2]
            max_x = width * self.analyzer.meters_per_pixel
            max_y = height * self.analyzer.meters_per_pixel
            return 0, max_x, 0, max_y
        return 0, 100, 0, 100
    
    def _prepare_axis_clean(self, ax: plt.Axes) -> plt.Axes:
        """
        Prepare matplotlib Axes for clean visualization (white background, no imagery).
        
        Configures axis properties for metrics rendered on white background:
        - Sets white background color
        - Establishes world coordinate limits encompassing entire ROI
        - Enforces 1:1 aspect ratio (nadir view assumption)
        - Adds axis labels indicating meter units
        
        Args:
            ax: Matplotlib Axes object to configure
        
        Returns:
            plt.Axes: Configured axis object
        """
        ax.set_facecolor('white')
        x_min, x_max, y_min, y_max = self._get_axis_limits()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Y inverted for geospatial mapping
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        return ax
    
    def _prepare_axis_overlay(self, ax: plt.Axes) -> plt.Axes:
        """
        Prepare matplotlib Axes for overlay visualization with orthomosaic background.
        
        Displays orthomosaic imagery with proper geospatial coordinate alignment. This is
        critical: the 'extent' parameter maps pixel space to world coordinates (meters) so
        that metric geometry (polygons, circles, lines) rendered in meter-space aligns
        correctly with the image background.
        
        BUG 1 FIX: Without extent parameter, meter-scale geometry appears as invisible
        dots in top-left corner of pixel-scale image. With extent, everything aligns.
        
        Args:
            ax: Matplotlib Axes object to configure
        
        Returns:
            plt.Axes: Configured axis with orthomosaic displayed
        
        Note:
            Falls back to clean white background if orthomosaic_array is None.
        """
        if self.orthomosaic_array is None:
            return self._prepare_axis_clean(ax)
        
        height, width = self.orthomosaic_array.shape[:2]
        max_x = width * self.analyzer.meters_per_pixel
        max_y = height * self.analyzer.meters_per_pixel
        extent = [0, max_x, max_y, 0]  # BUG 1 FIXED: Mapped pixels to physical space
        
        if len(self.orthomosaic_array.shape) == 2:
            ax.imshow(self.orthomosaic_array, cmap='gray', origin='upper', extent=extent)
        else:
            ax.imshow(self.orthomosaic_array, origin='upper', extent=extent)
        
        ax.set_xlim(0, max_x)
        ax.set_ylim(max_y, 0)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        return ax
    
    def _prepare_axis_3d(self, ax: plt.Axes, elev: float = 30, azim: float = -60) -> plt.Axes:
        """
        Prepare matplotlib Axes for 3D visualization with proper scaling and viewing angles.
        
        Configures 3D axis properties to prevent vertical exaggeration artifacts and
        ensure consistent perspective across all 3D figures. Matplotlib's set_aspect('equal')
        is unreliable in 3D space, so we manually compute box aspect ratio.
        
        Args:
            ax: Matplotlib Axes3D object (must be created with projection='3d')
            elev: Viewing elevation angle in degrees (default: 30° above horizon)
            azim: Viewing azimuth angle in degrees (default: -60° from north)
        
        Returns:
            plt.Axes: Configured 3D axis object
        
        Note:
            Z-axis is compressed to 30% of XY range to make topography features visible
            without absurd vertical spiking. This is a standard practice for terrain viz.
        """
        ax.set_facecolor('white')
        ax.view_init(elev=elev, azim=azim)
        
        x_min, x_max, y_min, y_max = self._get_axis_limits()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Invert Y for geospatial mapping
        
        # Compute box aspect ratio to balance X, Y, Z dimensions
        # Without this, matplotlib either compresses Z to invisibility or exaggerates it
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_compression = 0.3  # Z-axis = 30% of XY range for visual clarity
        ax.set_box_aspect((x_range, y_range, min(x_range, y_range) * z_compression))
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Elevation (meters)')
        return ax
    
    def _calculate_safe_stride(self, array_shape: Tuple[int, int], max_dimension: int = 1500) -> int:
        """
        Calculate fail-safe decimation stride to prevent 3D rendering crashes.
        
        Matplotlib's mplot3d module is notorious for freezing on large meshes. This method
        computes the minimum stride needed to ensure the decimated array never exceeds
        max_dimension pixels on its longest edge.
        
        Args:
            array_shape: Shape of elevation array (height, width)
            max_dimension: Maximum allowed pixels per edge after decimation (default: 1500)
        
        Returns:
            int: Stride value (minimum 1) to use in array[::stride, ::stride]
        """
        max_edge = max(array_shape)
        stride = max(1, int(np.ceil(max_edge / max_dimension)))
        return stride
    
    def _render_and_save(self, metric_name: str, plot_func: Callable[[plt.Axes, str], None]) -> Dict[str, str]:
        """
        Master rendering engine: generate and save all three figure variants for a metric.
        
        Executes a metric-specific plotting function three times to produce:
        1. Standalone clean figure: metric on white background
        2. Standalone overlay figure: metric on orthomosaic imagery  
        3. Combined 1x2 subplot: clean and overlay side-by-side for comparison
        
        This eliminates code duplication by accepting a generic plotting callback that
        handles both 'clean' and 'overlay' modes.
        
        Args:
            metric_name: Name of the metric for filename generation (e.g., 'nnd', 'passability')
            plot_func: Callback function with signature plot_func(ax, mode: str) where
                       mode is either 'clean' or 'overlay'. Function should populate axis.
        
        Returns:
            Dict[str, str]: Mapping of output types to file paths:
                {'clean': 'path/to/metric_clean.png',
                 'overlay': 'path/to/metric_overlay.png',
                 'combined': 'path/to/metric_combined.png'}
        """
        output_paths = {}
        
        # 1. STANDALONE CLEAN
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.figure_dpi)
        self._prepare_axis_clean(ax)
        plot_func(ax, 'clean')
        clean_path = self.output_dir / f"{metric_name}_clean.png"
        plt.savefig(clean_path, dpi=self.figure_dpi, bbox_inches='tight')
        output_paths['clean'] = str(clean_path)
        plt.close(fig)
        
        # 2. STANDALONE OVERLAY
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.figure_dpi)
        self._prepare_axis_overlay(ax)
        plot_func(ax, 'overlay')
        overlay_path = self.output_dir / f"{metric_name}_overlay.png"
        plt.savefig(overlay_path, dpi=self.figure_dpi, bbox_inches='tight')
        output_paths['overlay'] = str(overlay_path)
        plt.close(fig)
        
        # 3. COMBINED SUBPLOT (1x2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figure_size[0]*2, self.figure_size[1]), dpi=self.figure_dpi)
        self._prepare_axis_clean(ax1)
        plot_func(ax1, 'clean')
        ax1.set_title(f'{metric_name} (Clean)', fontsize=14, fontweight='bold')
        
        self._prepare_axis_overlay(ax2)
        plot_func(ax2, 'overlay')
        ax2.set_title(f'{metric_name} (Overlay)', fontsize=14, fontweight='bold')
        
        combined_path = self.output_dir / f"{metric_name}_combined.png"
        plt.savefig(combined_path, dpi=self.figure_dpi, bbox_inches='tight')
        output_paths['combined'] = str(combined_path)
        plt.close(fig)
        
        return output_paths
    
    def _render_and_save_3d_suite(
        self,
        metric_name: str,
        plot_func_2d: Callable[[plt.Axes, str], None],
        plot_func_3d: Callable[[plt.Axes, str], None]
    ) -> Dict[str, str]:
        """
        The 'Rule of Five' rendering engine for verticality metrics.
        
        Executes metric-specific plotting functions to produce five output variants:
        1. 2D Clean: Flat map view on white background
        2. 2D Overlay: Flat map view over orthomosaic imagery
        3. 2D Combined: 1x2 subplot comparison of clean and overlay
        4. 3D Clean: Topographic mesh with color-coded elevation
        5. 3D Overlay: Photo-realistic RGB texture draped over 3D terrain
        
        This architecture extends the 2D "Rule of Three" pattern to include 3D physical
        models, providing both quick-scan map views and intuitive terrain verification.
        
        Args:
            metric_name: Name of metric for filename generation (e.g., 'protrusion')
            plot_func_2d: Callback for 2D plotting with signature func(ax, mode: str)
            plot_func_3d: Callback for 3D plotting with signature func(ax, mode: str)
        
        Returns:
            Dict[str, str]: Mapping of output types to file paths:
                {'2d_clean': ..., '2d_overlay': ..., '2d_combined': ...,
                 '3d_clean': ..., '3d_overlay': ...}
        
        Note:
            If elevation_array is None, only the 2D suite is generated and 3D is skipped
            with informational message (graceful degradation for datasets without DEMs).
        """
        output_paths = {}
        
        # STEP 1-3: Generate standard 2D suite using existing master function
        print(f"[RENDER] Generating 2D suite for {metric_name}...")
        paths_2d = self._render_and_save(metric_name, plot_func_2d)
        output_paths.update({f"2d_{k}": v for k, v in paths_2d.items()})
        
        # Check if elevation data is available for 3D rendering
        if self.elevation_array is None:
            print(f"[INFO] Skipping 3D renders for {metric_name} - No elevation data.")
            return output_paths
        
        # STEP 4: STANDALONE 3D CLEAN
        print(f"[RENDER] Generating 3D clean figure for {metric_name}...")
        fig = plt.figure(figsize=self.figure_size, dpi=self.figure_dpi)
        ax = fig.add_subplot(111, projection='3d')
        self._prepare_axis_3d(ax)
        try:
            plot_func_3d(ax, 'clean')
            clean_3d_path = self.output_dir / f"{metric_name}_3d_clean.png"
            plt.savefig(clean_3d_path, dpi=self.figure_dpi, bbox_inches='tight')
            output_paths['3d_clean'] = str(clean_3d_path)
            print(f"  ✓ Saved: {clean_3d_path.name}")
        except Exception as e:
            print(f"  ✗ Error in 3D Clean: {e}")
        finally:
            plt.close(fig)
        
        # STEP 5: STANDALONE 3D OVERLAY
        print(f"[RENDER] Generating 3D overlay figure for {metric_name}...")
        fig = plt.figure(figsize=self.figure_size, dpi=self.figure_dpi)
        ax = fig.add_subplot(111, projection='3d')
        self._prepare_axis_3d(ax)
        try:
            plot_func_3d(ax, 'overlay')
            overlay_3d_path = self.output_dir / f"{metric_name}_3d_overlay.png"
            plt.savefig(overlay_3d_path, dpi=self.figure_dpi, bbox_inches='tight')
            output_paths['3d_overlay'] = str(overlay_3d_path)
            print(f"  ✓ Saved: {overlay_3d_path.name}")
        except Exception as e:
            print(f"  ✗ Error in 3D Overlay: {e}")
        finally:
            plt.close(fig)
        
        return output_paths

    def visualize_nearest_neighbor_distance(self, distance_threshold_m: float = 0.05) -> Dict[str, str]:
        """
        Visualize nearest neighbor distances to identify jamming risk areas.
        
        Generates three figures showing physical gaps between polygons (minerals/nodules).
        Small gaps are highlighted in red to identify high-risk regions where a collection
        vehicle might jam. This is critical for deep-sea mining operations.
        
        Figures:
        - Clean: Gray polygons with colored distance lines
        - Overlay: Same visualization over seafloor orthomosaic
        - Combined: 1x2 subplot for direct comparison
        
        Args:
            distance_threshold_m: Distance threshold in meters below which gap is considered
                                  high-risk (default: 0.05 = 5cm, typical collector mesh width)
        
        Returns:
            Dict[str, str]: Output file paths {'clean': ..., 'overlay': ..., 'combined': ...}
        """
        metric_results = self.analyzer.calculate_nearest_neighbor_distance(method='edge')
        
        def plot_nnd(ax: plt.Axes, mode: str) -> None:
            """Inner function: render NND visualization for specified mode (clean or overlay)."""
            polygons = self.analyzer._load_polygons()
            
            # Draw polygons
            for poly in polygons:
                coords = np.array(poly.exterior.coords) * self.analyzer.meters_per_pixel
                if mode == 'clean':
                    patch = mpatches.Polygon(coords, fill=True, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7)
                else:
                    patch = mpatches.Polygon(coords, fill=False, edgecolor='white', linewidth=1.0, alpha=0.5)
                ax.add_patch(patch)
            
            # Draw nearest neighbor distance lines
            if 'nnd_values_m' in metric_results:
                # Simplified pairing logic for visualizer (assuming points are close enough for demonstration)
                for i, poly1 in enumerate(polygons):
                    min_dist = float('inf')
                    closest_points = None
                    exterior1 = np.array(poly1.exterior.coords) * self.analyzer.meters_per_pixel
                    
                    for j, poly2 in enumerate(polygons):
                        if i == j: 
                            continue
                        exterior2 = np.array(poly2.exterior.coords) * self.analyzer.meters_per_pixel
                        # Find closest points between these two exteriors
                        for pt1 in exterior1:
                            for pt2 in exterior2:
                                d = np.linalg.norm(pt1 - pt2)
                                if d < min_dist:
                                    min_dist = d
                                    closest_points = (pt1, pt2)
                                    
                    if closest_points is not None:
                        pt1, pt2 = closest_points
                        distance = min_dist
                        
                        color = '#FF0000' if distance < distance_threshold_m else '#FFB800'
                        linewidth = 2.0 if distance < distance_threshold_m else 1.0
                        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=linewidth, zorder=5)
            
            red_line = plt.Line2D([0], [0], color='#FF0000', linewidth=2, 
                                  label=f'High risk (< {distance_threshold_m*100:.1f}cm)')
            yellow_line = plt.Line2D([0], [0], color='#FFB800', linewidth=1.0, label='Moderate risk')
            ax.legend(handles=[red_line, yellow_line], loc='upper right')
        
        return self._render_and_save('nearest_neighbor_distance', plot_nnd)

    def visualize_passability_index(self) -> Dict[str, str]:
        """
        Visualize passability index (navigable corridors for collection vehicles).
        
        Generates three figures showing where collection vehicles can navigate without
        colliding with objects. Displays a 2D distance transform heatmap (distance to
        nearest obstacle) with the maximum inscribed circle highlighted in red.
        
        Circle is now drawn at exact location of largest passage, not at
        random center position. Location is calculated automatically from distance transform.
        
        Figures:
        - Clean: Distance heatmap with colorbar and maximum circle
        - Overlay: Same heatmap over orthomosaic (semi-transparent)
        - Combined: 1x2 subplot for comparison
        
        Returns:
            Dict[str, str]: Output file paths {'clean': ..., 'overlay': ..., 'combined': ...}
        """
        metric_results = self.analyzer.calculate_passability_index()
        
        def plot_passability(ax: plt.Axes, mode: str) -> None:
            """Inner function: render passability visualization for specified mode."""
            mask = self.analyzer._load_mask()
            # Substrate (mask == 1) is free space for navigation
            # Mask structure: 0=background, 1=substrate, 2=nodule, 3=organisms
            sediment_mask = (mask == 1).astype(np.uint8)
            distance_transform = ndimage.distance_transform_edt(sediment_mask)
            distance_transform_m = distance_transform * self.analyzer.meters_per_pixel
            
            height, width = mask.shape
            extent = [0, width * self.analyzer.meters_per_pixel, height * self.analyzer.meters_per_pixel, 0]
            
            alpha = 1.0 if mode == 'clean' else 0.5
            im = ax.imshow(distance_transform_m, cmap='viridis', origin='upper', extent=extent, alpha=alpha)
            
            if mode == 'clean':
                plt.colorbar(im, ax=ax, label='Distance to Obstacle (meters)')
            
            if 'max_passage_radius_m' in metric_results and metric_results['max_passage_radius_m'] > 0:
                # Prefer computing the radius/location from the displayed distance_transform to
                # guarantee unit consistency with the imshow'd raster (distance_transform_m).
                max_idx = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)
                max_y = max_idx[0] * self.analyzer.meters_per_pixel
                max_x = max_idx[1] * self.analyzer.meters_per_pixel
                
                # Compute radius from the displayed map (meters). This avoids unit mismatch
                # if the analyzer returned a pixel-valued radius by mistake.
                max_radius_m = float(distance_transform_m.max())
                
                # If analyzer also provided a radius, log a small-warning overlay (visual check)
                supplied = metric_results.get('max_passage_radius_m', None)
                if supplied is not None:
                    supplied = float(supplied)
                    # If supplied value differs substantially, prefer the computed value but draw both lightly.
                    if abs(supplied - max_radius_m) / (max_radius_m + 1e-9) > 0.2:
                        # draw supplied as a thin dashed circle for debugging
                        debug_circle = mpatches.Circle((max_x, max_y), supplied, fill=False, edgecolor='orange', linewidth=1.0, linestyle='--', alpha=0.6)
                        ax.add_patch(debug_circle)
                
                if max_radius_m > 0:
                    circle = mpatches.Circle((max_x, max_y), max_radius_m, fill=False, edgecolor='#FF0000', linewidth=3, zorder=10)
                    ax.add_patch(circle)
                    ax.text(max_x, max_y - max_radius_m - 0.5, f'Max radius: {max_radius_m:.2f}m',
                           ha='center', fontsize=12, color='#FF0000', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=11)
        
        return self._render_and_save('passability_index', plot_passability)

    def visualize_solidity_rugosity(self, n_exemplars: int = 4, padding_fraction: float = 0.2) -> Dict[str, str]:
        """
        Visualize polygon solidity/rugosity through exemplar zoom-in grid.
        
        Generates grid figures showing selected exemplar polygons across the solidity
        spectrum (smooth to jagged). Each cell displays:
        - Gray polygon
        - Green convex hull overlay (to illustrate the "rubber band" concept of solidity)
        
        BUG 3 FIX: Completely separate rendering path. Each figure (clean, overlay,
        combined) is generated independently with proper grid layout.
        
        Figures:
        - Clean: 2x2 grid of exemplars on white background
        - Overlay: 2x2 grid of exemplars on orthomosaic crops
        - Combined: Nx2 grid with clean and overlay side-by-side for each exemplar
        
        Args:
            n_exemplars: Number of exemplar polygons to display (default: 4)
            padding_fraction: Padding around polygon bounding box as fraction of bbox size
        
        Returns:
            Dict[str, str]: Output file paths {'clean': ..., 'overlay': ..., 'combined': ...}
        """
        morphology_df = self.analyzer.calculate_morphology_stats()
        if len(morphology_df) == 0: return {}
        
        # Select two lowest-solidity and two highest-solidity exemplars (preserve order).
        # If the dataframe is smaller than 4, choose as many unique entries as available.
        sorted_df = morphology_df.sort_values('solidity')
        n = len(sorted_df)
        indices = []
        # Two lowest
        for i in range(min(2, n)):
            indices.append(sorted_df.index[i])
        # Two highest
        for i in range(min(2, n)):
            indices.append(sorted_df.index[n - 1 - i])
        # Preserve order and deduplicate
        unique_indices = list(dict.fromkeys(indices))
        # Respect n_exemplars if provided (optional cap)
        if n_exemplars is not None and n_exemplars > 0:
            unique_indices = unique_indices[:n_exemplars]
        self._exemplar_indices = unique_indices
        polygons = self.analyzer._load_polygons()
        
        def draw_cell(ax, poly, solidity, mode):
            """
            Draw a single solidity exemplar cell showing the polygon and convex hull.
            
            Coordinates: polygon bounds are in pixel space, converted to meters for display.
            For overlay mode, extracts and displays the orthomosaic crop for that region.
            """
            # Get polygon bounds in pixel space (original raster coordinates)
            px_min, py_min, px_max, py_max = poly.bounds
            width_px = px_max - px_min
            height_px = py_max - py_min
            # Apply padding in pixel space
            pad_px = width_px * padding_fraction
            pad_py = height_px * padding_fraction
            crop_px_min = int(px_min - pad_px)
            crop_py_min = int(py_min - pad_py)
            crop_px_max = int(px_max + pad_px)
            crop_py_max = int(py_max + pad_py)
            
            # Clamp to image bounds to avoid index errors
            if self.orthomosaic_array is not None:
                img_height, img_width = self.orthomosaic_array.shape[:2]
                crop_px_min = max(0, crop_px_min)
                crop_py_min = max(0, crop_py_min)
                crop_px_max = min(img_width, crop_px_max)
                crop_py_max = min(img_height, crop_py_max)
            
            # Convert crop pixel bounds to world coordinates
            crop_world_xmin = crop_px_min * self.analyzer.meters_per_pixel
            crop_world_ymin = crop_py_min * self.analyzer.meters_per_pixel
            crop_world_xmax = crop_px_max * self.analyzer.meters_per_pixel
            crop_world_ymax = crop_py_max * self.analyzer.meters_per_pixel
            
            if mode == 'clean':
                ax.set_facecolor('white')
                
            elif mode == 'overlay' and self.orthomosaic_array is not None:
                # Extract the orthomosaic crop in pixel space
                img_crop = self.orthomosaic_array[crop_py_min:crop_py_max, crop_px_min:crop_px_max]
                
                if img_crop.size > 0:
                    # Display crop with extent mapping pixels to world coordinates
                    extent = [crop_world_xmin, crop_world_xmax, crop_world_ymax, crop_world_ymin]
                    ax.imshow(img_crop, origin='upper', extent=extent)
            
            # Draw polygon filled in clean mode, outline only in overlay mode
            poly_coords_m = np.array(poly.exterior.coords) * self.analyzer.meters_per_pixel
            if mode == 'clean':
                ax.add_patch(mpatches.Polygon(poly_coords_m, fill=True, facecolor='gray', 
                                              edgecolor='black', linewidth=0.5, alpha=0.8))
            else:
                # In overlay mode, still show polygon outline for reference
                ax.add_patch(mpatches.Polygon(poly_coords_m, fill=False, edgecolor='white', 
                                              linewidth=1.0, alpha=0.7))
            
            # Draw convex hull in bright neon green (always visible)
            hull = poly.convex_hull
            if hull.geom_type == 'Polygon':
                hull_coords_m = np.array(hull.exterior.coords) * self.analyzer.meters_per_pixel
                ax.add_patch(mpatches.Polygon(hull_coords_m, fill=False, edgecolor='#00FF00', 
                                              linewidth=2.5, alpha=0.9))
            
            # Set axis limits to show the padded crop region in world coordinates
            ax.set_xlim(crop_world_xmin, crop_world_xmax)
            ax.set_ylim(crop_world_ymax, crop_world_ymin)  # Y inverted for image space
            ax.set_aspect('equal')
            ax.set_title(f'Solidity: {solidity:.3f}', fontsize=12, fontweight='bold')
            ax.axis('off')

        output_paths = {}
        
        # 1. Standalone Clean Grid
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.figure_dpi)
        for ax, idx in zip(axes.flatten(), self._exemplar_indices):
            draw_cell(ax, polygons[idx], morphology_df.iloc[idx]['solidity'], 'clean')
        plt.tight_layout()
        clean_path = self.output_dir / "solidity_rugosity_clean.png"
        plt.savefig(clean_path, dpi=self.figure_dpi)
        output_paths['clean'] = str(clean_path)
        plt.close(fig)

        # 2. Standalone Overlay Grid
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.figure_dpi)
        for ax, idx in zip(axes.flatten(), self._exemplar_indices):
            draw_cell(ax, polygons[idx], morphology_df.iloc[idx]['solidity'], 'overlay')
        plt.tight_layout()
        overlay_path = self.output_dir / "solidity_rugosity_overlay.png"
        plt.savefig(overlay_path, dpi=self.figure_dpi)
        output_paths['overlay'] = str(overlay_path)
        plt.close(fig)

        # 3. Combined Grid (Rows = Exemplars, Col1 = Clean, Col2 = Overlay)
        fig, axes = plt.subplots(len(self._exemplar_indices), 
                                 2, 
                                 figsize=(10, 4 * len(self._exemplar_indices)), 
                                 dpi=self.figure_dpi)
        
        for row_idx, idx in enumerate(self._exemplar_indices):
            draw_cell(axes[row_idx, 0], polygons[idx], morphology_df.iloc[idx]['solidity'], 'clean')
            draw_cell(axes[row_idx, 1], polygons[idx], morphology_df.iloc[idx]['solidity'], 'overlay')
        
        plt.tight_layout()
        combined_path = self.output_dir / "solidity_rugosity_combined.png"
        plt.savefig(combined_path, dpi=self.figure_dpi)
        output_paths['combined'] = str(combined_path)
        plt.close(fig)

        return output_paths
    
    # =========================================================================
    # PHASE 3: VERTICALITY METRICS (3D - REQUIRE ELEVATION)
    # =========================================================================
    
    def visualize_protrusion(self, stride: Optional[int] = None) -> Dict[str, str]:
        """
        Visualize Protrusion (Stick-up Height) - The Collector Clearance Metric.
        
        Generates 2D heatmaps and 3D topographic models showing vertical exposure of
        objects above the interpolated seafloor baseline. This metric directly informs
        mining vehicle collector head height settings.
        
        Creates five figure variants (Rule of Five):
        - 2D Clean: Heatmap on white background (plasma colormap: dark=flush, bright=tall)
        - 2D Overlay: Semi-transparent heatmap over orthomosaic photo
        - 2D Combined: 1x2 subplot comparison
        - 3D Clean: Topographic mesh with semi-transparent "virtual seafloor plane"
        - 3D Overlay: Photo-realistic RGB texture draped over terrain with glass plane
        
        Args:
            stride: Decimation factor for 3D meshes (plot every Nth pixel).
                   If None, automatically calculated to prevent memory crashes.
                   Recommended: 5-10 for datasets <5000px, 20+ for larger datasets.
        
        Returns:
            Dict[str, str]: Output file paths for all five figures
        
        Raises:
            ValueError: If analyzer has not computed protrusion metric (call calculate_protrusion first)
        
        Example:
            >>> viz = SpatialMetricsVisualizer(analyzer, output_dir="figs/")
            >>> paths = viz.visualize_protrusion(stride=10)
            >>> print(f"3D overlay: {paths['3d_overlay']}")
        
        Note:
            - If mean stick-up is ~2cm, collector must be lowered aggressively
            - If stick-up is ~10cm, collector can "clip" nodules off the top cleanly
            - Elevation raster MUST be in METERS matching XY scale (not mm or raw acoustic)
        """
        print("\n" + "="*60)
        print("VISUALIZING PROTRUSION (STICK-UP HEIGHT)")
        print("="*60)
        
        # Fetch metric results from analyzer
        metric_results = self.analyzer.calculate_protrusion()
        seafloor_z = metric_results.get('seafloor_elevation_m', 0.0)
        
        # Load required data arrays
        # Get mask to determine target dimensions
        mask = self.analyzer._load_mask()
        
        # Load elevation (may be different resolution)
        elevation_raw = self.elevation_array if self.elevation_array is not None else self.analyzer._load_elevation()
        
        # Resample elevation to match mask dimensions for pixel-wise operations
        elevation = self.analyzer._resample_elevation_to_mask(elevation_raw, mask)
        
        meters_per_px = self.analyzer.meters_per_pixel
        
        # Calculate smart default stride if not provided
        if stride is None:
            stride = self._calculate_safe_stride(elevation.shape)
            print(f"[INFO] Auto-calculated stride={stride} for safe 3D rendering")
        
        # --- 2D PLOTTING LOGIC ---
        def plot_protrusion_2d(ax: plt.Axes, mode: str) -> None:
            """
            Render 2D protrusion heatmap.
            
            Clean mode: Full opacity plasma colormap on white background
            Overlay mode: 60% opacity plasma overlay on orthomosaic
            """
            # Calculate stick-up: object elevation minus virtual seafloor plane
            stick_up = elevation - seafloor_z
            
            # Mask out sediment (mask==1); only objects (mask>=2) get colors
            # Background (mask==0) and sediment become NaN (transparent white)
            stick_up_masked = np.where(mask >= 2, stick_up, np.nan)
            
            # Map pixel coordinates to physical meters
            extent = [0, mask.shape[1] * meters_per_px, mask.shape[0] * meters_per_px, 0]
            
            # Overlay mode uses transparency to show underlying seafloor texture
            alpha = 1.0 if mode == 'clean' else 0.6
            
            # Plasma colormap: dark purple=flush with mud, bright yellow=tall protrusion
            im = ax.imshow(
                stick_up_masked,
                cmap='plasma',
                origin='upper',
                extent=extent,
                alpha=alpha,
                vmin=0,  # Color scale starts at zero stick-up
                vmax=np.nanmax(stick_up_masked)
            )
            
            # Only show colorbar in clean mode to avoid clutter
            if mode == 'clean':
                cbar = plt.colorbar(im, ax=ax, label='Stick-up Height (meters)', shrink=0.8)
                cbar.ax.tick_params(labelsize=9)
            
            ax.set_title('Protrusion: Collector Clearance Map', fontsize=12, fontweight='bold')
        
        # --- 3D PLOTTING LOGIC ---
        def plot_protrusion_3d(ax: plt.Axes, mode: str) -> None:
            """
            Render 3D protrusion with semi-transparent seafloor plane.
            
            Clean mode: Terrain colormap showing topographic elevation
            Overlay mode: Photo-realistic RGB texture draped over 3D mesh
            
            Both modes include blue glass "virtual seafloor plane" for visual reference.
            """
            # Downsample elevation array to prevent memory crashes
            elev_dec = elevation[::stride, ::stride]
            
            # Create meshgrid in world coordinates (meters)
            x = np.linspace(0, mask.shape[1] * meters_per_px, elev_dec.shape[1])
            y = np.linspace(0, mask.shape[0] * meters_per_px, elev_dec.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # 1. RENDER VIRTUAL SEAFLOOR PLANE (semi-transparent blue glass)
            Z_seafloor = np.full_like(elev_dec, seafloor_z)
            ax.plot_surface(
                X, Y, Z_seafloor,
                color='#00aaff',
                alpha=0.3,
                linewidth=0,
                antialiased=False,
                zorder=1  # Render behind topography
            )
            
            # 2. RENDER TOPOGRAPHIC MESH
            if mode == 'clean':
                # Clean mode: Color by raw elevation using terrain colormap
                ax.plot_surface(
                    X, Y, elev_dec,
                    cmap='terrain',
                    linewidth=0,
                    antialiased=True,
                    alpha=0.9,
                    zorder=2
                )
            
            elif mode == 'overlay' and self.orthomosaic_array is not None:
                # Overlay mode: Drape RGB orthomosaic as texture over topography
                # This preserves natural shadows and seafloor features
                img_dec = self.orthomosaic_array[::stride, ::stride]
                
                # CRITICAL: shade=False prevents double-shadowing
                # (orthomosaic already contains ROV strobe shadows)
                ax.plot_surface(
                    X, Y, elev_dec,
                    facecolors=img_dec,
                    linewidth=0,
                    antialiased=True,
                    shade=False,  # Preserve baked-in photography lighting
                    zorder=2
                )
            
            ax.set_title('3D Protrusion & Virtual Seafloor Plane', fontsize=12, fontweight='bold')
        
        # Execute the Rule of Five rendering pipeline
        return self._render_and_save_3d_suite('protrusion', plot_protrusion_2d, plot_protrusion_3d)
    
    def visualize_embedment_angle(self, stride: Optional[int] = None) -> Dict[str, str]:
        """
        Visualize Embedment Angle (Contact Slope) - The Breakout Force Metric.
        
        Generates 2D perimeter ring maps and 3D collar visualizations showing the steepness
        of the slope where nodules contact the sediment. Steep angles indicate objects sitting
        loosely on top (easy extraction), shallow angles indicate burial/draping (high breakout
        force and sediment disturbance).
        
        Creates five figure variants (Rule of Five):
        - 2D Clean: Colored perimeter rings on white (red=shallow/buried, green=steep/loose)
        - 2D Overlay: Perimeter rings overlaid on orthomosaic photo
        - 2D Combined: 1x2 subplot comparison
        - 3D Clean: Topographic mesh colored by surface slope gradient
        - 3D Overlay: RGB texture with 3D scatter "collar" at nodule bases colored by angle
        
        Args:
            stride: Decimation factor for 3D meshes. If None, auto-calculated for safety.
        
        Returns:
            Dict[str, str]: Output file paths for all five figures
        
        Example:
            >>> viz = SpatialMetricsVisualizer(analyzer, output_dir="figs/")
            >>> paths = viz.visualize_embedment_angle()
            >>> print(f"Overlay shows red rings on deeply embedded nodules")
        
        Note:
            - Red rings: Shallow angle, nodule draped/buried, requires high suction force
            - Green rings: Steep angle, nodule sitting on top, easy mechanical pickup
            - This metric predicts turbidity plume generation during extraction
        """
        print("\n" + "="*60)
        print("VISUALIZING EMBEDMENT ANGLE (CONTACT SLOPE)")
        print("="*60)
        
        # Load required data
        # Get mask to determine target dimensions
        mask = self.analyzer._load_mask()
        
        # Load elevation (may be different resolution)
        elevation_raw = self.elevation_array if self.elevation_array is not None else self.analyzer._load_elevation()
        
        # Resample elevation to match mask dimensions for pixel-wise operations
        elevation = self.analyzer._resample_elevation_to_mask(elevation_raw, mask)
        
        meters_per_px = self.analyzer.meters_per_pixel
        
        # Calculate smart default stride if not provided
        if stride is None:
            stride = self._calculate_safe_stride(elevation.shape)
            print(f"[INFO] Auto-calculated stride={stride} for safe 3D rendering")
        
        # Extract perimeter ring using morphological erosion
        # Perimeter = dilated mask minus original mask
        eroded_mask = ndimage.binary_erosion(mask >= 2).astype(mask.dtype)
        perimeter_mask = ((mask >= 2).astype(int) - eroded_mask) > 0
        
        if not np.any(perimeter_mask):
            print("[WARNING] No perimeter pixels found (objects too small or mask invalid)")
            return {}
        
        # Calculate terrain gradient (slope)
        dz_dy, dz_dx = np.gradient(elevation * meters_per_px)  # Scale to meters
        slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_degrees = np.degrees(slope_radians)
        
        # --- 2D PLOTTING LOGIC ---
        def plot_embedment_2d(ax: plt.Axes, mode: str) -> None:
            """
            Render 2D embedment angle as colored perimeter rings.
            
            RdYlGn_r colormap: Red=shallow/embedded (bad), Green=steep/loose (good)
            """
            # Isolate slope values at perimeter pixels only
            perimeter_slopes = np.where(perimeter_mask, slope_degrees, np.nan)
            
            extent = [0, mask.shape[1] * meters_per_px, mask.shape[0] * meters_per_pixel, 0]
            
            # Reversed Red-Yellow-Green: intuitive traffic light metaphor
            im = ax.imshow(
                perimeter_slopes,
                cmap='RdYlGn_r',
                origin='upper',
                extent=extent,
                alpha=1.0,
                vmin=0,
                vmax=90  # Slope angles range 0-90 degrees
            )
            
            if mode == 'clean':
                cbar = plt.colorbar(im, ax=ax, label='Contact Slope / Embedment Angle (°)', shrink=0.8)
                cbar.ax.tick_params(labelsize=9)
            
            ax.set_title('Embedment Angle: Breakout Force Perimeter Rings', fontsize=12, fontweight='bold')
        
        # --- 3D PLOTTING LOGIC ---
        def plot_embedment_3d(ax: plt.Axes, mode: str) -> None:
            """
            Render 3D embedment angle visualization.
            
            Clean mode: Mesh colored by surface gradient
            Overlay mode: RGB texture with 3D scatter collar at perimeter
            """
            elev_dec = elevation[::stride, ::stride]
            x = np.linspace(0, mask.shape[1] * meters_per_px, elev_dec.shape[1])
            y = np.linspace(0, mask.shape[0] * meters_per_px, elev_dec.shape[0])
            X, Y = np.meshgrid(x, y)
            
            if mode == 'clean':
                # Clean mode: Surface colored strictly by slope gradient
                slope_dec = slope_degrees[::stride, ::stride]
                
                # Normalize slope to colormap range [0, 90 degrees]
                norm = plt.Normalize(0, 90)
                colors = plt.cm.RdYlGn_r(norm(slope_dec))
                
                ax.plot_surface(
                    X, Y, elev_dec,
                    facecolors=colors,
                    linewidth=0,
                    antialiased=True,
                    alpha=0.9
                )
                
                # Create fake mappable for colorbar
                m = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
                m.set_array([])
                plt.colorbar(m, ax=ax, label='Surface Gradient (°)', shrink=0.5)
            
            elif mode == 'overlay' and self.orthomosaic_array is not None:
                # 1. Drape RGB photo over terrain (base layer)
                img_dec = self.orthomosaic_array[::stride, ::stride]
                ax.plot_surface(
                    X, Y, elev_dec,
                    facecolors=img_dec,
                    linewidth=0,
                    antialiased=True,
                    shade=False,  # Preserve natural lighting
                    alpha=0.8,
                    zorder=1
                )
                
                # 2. Draw 3D "collar" as scatter points at perimeter
                # Extract perimeter pixel coordinates
                py, px = np.where(perimeter_mask)
                pz = elevation[py, px]
                pslopes = slope_degrees[py, px]
                
                # Convert to physical world coordinates (meters)
                px_m = px * meters_per_px
                py_m = py * meters_per_px
                
                # Filter out NaN elevations/slopes
                valid = ~(np.isnan(pz) | np.isnan(pslopes))
                px_m, py_m, pz, pslopes = px_m[valid], py_m[valid], pz[valid], pslopes[valid]
                
                # Render as 3D scatter forming glowing collar around nodule bases
                # Marker size: adjust based on raster resolution (s=5 for high-res, s=20 for coarse)
                scatter = ax.scatter3D(
                    px_m, py_m, pz,
                    c=pslopes,
                    cmap='RdYlGn_r',
                    s=5,  # Marker size in points
                    alpha=1.0,
                    depthshade=False,  # Disable depth-based shading for consistent color
                    vmin=0,
                    vmax=90,
                    zorder=10  # Render on top of terrain
                )
                plt.colorbar(scatter, ax=ax, label='Contact Angle (°)', shrink=0.5)
            
            ax.set_title('3D Embedment Angle', fontsize=12, fontweight='bold')
        
        # Execute the Rule of Five rendering pipeline
        return self._render_and_save_3d_suite('embedment_angle', plot_embedment_2d, plot_embedment_3d)

    def visualize_obb_directionality(self) -> Dict[str, str]:
        """
        Visualize Oriented Bounding Box angles to reveal paleo-current direction.
        
        Generates three figures showing the principal axis orientation of elongated polygons.
        Angle is encoded in color (HSV cyclic colormap: 0°→red/east, 90°→green/north).
        Useful for identifying current-driven alignment patterns in deep-sea deposits.
        
        BUG 4 FIX: Clean view now shows gray polygon context before drawing directional
        lines. Without polygons, lines appeared meaningless and random.
        
        Figures:
        - Clean: Gray polygons with colored directional lines through centroids
        - Overlay: Translucent OBB rectangles colored by angle orientation
        - Combined: 1x2 subplot showing both views
        
        Returns:
            Dict[str, str]: Output file paths {'clean': ..., 'overlay': ..., 'combined': ...}
        """
        morphology_df = self.analyzer.calculate_morphology_stats()
        
        def plot_obb(ax: plt.Axes, mode: str) -> None:
            """Inner function: render OBB visualization for specified mode."""
            polygons = self.analyzer._load_polygons()
            lines, colors = [], []
            
            # BUG 4 FIXED: Draw the underlying gray polygons in clean mode for context
            if mode == 'clean':
                for poly in polygons:
                    coords = np.array(poly.exterior.coords) * self.analyzer.meters_per_pixel
                    patch = mpatches.Polygon(coords, 
                                             fill=True, 
                                             facecolor='gray', 
                                             edgecolor='black', 
                                             linewidth=0.5, 
                                             alpha=0.5)
                    ax.add_patch(patch)
            
            for i, poly in enumerate(polygons):
                if i >= len(morphology_df): continue
                
                centroid_x = morphology_df.iloc[i]['centroid_x_m']
                centroid_y = morphology_df.iloc[i]['centroid_y_m']
                
                obb = poly.minimum_rotated_rectangle
                if obb.geom_type != 'Polygon': 
                    continue
                obb_coords = np.array(obb.exterior.coords)
                
                side1_vec = obb_coords[1] - obb_coords[0]
                side2_vec = obb_coords[2] - obb_coords[1]
                principal_vec = side1_vec if np.linalg.norm(side1_vec) > np.linalg.norm(side2_vec) else side2_vec
                principal_vec = principal_vec / (np.linalg.norm(principal_vec) + 1e-6)
                
                angle_rad = np.arctan2(principal_vec[1], principal_vec[0])
                angle_normalized = (np.degrees(angle_rad) % 360) / 360.0
                color = hsv_to_rgb([angle_normalized, 1.0, 1.0])
                
                if mode == 'clean':
                    line_length = 0.5 
                    start = np.array([centroid_x, centroid_y]) - principal_vec * line_length / 2
                    end = np.array([centroid_x, centroid_y]) + principal_vec * line_length / 2
                    lines.append([start, end])
                    colors.append(color)
                
                elif mode == 'overlay':
                    obb_coords_scaled = obb_coords * self.analyzer.meters_per_pixel
                    patch = mpatches.Polygon(obb_coords_scaled, 
                                             fill=True, 
                                             facecolor=color, 
                                             edgecolor=color, 
                                             linewidth=1.5, 
                                             alpha=0.3)
                    ax.add_patch(patch)
                    ax.plot(obb_coords_scaled[:, 0], obb_coords_scaled[:, 1], color=color, linewidth=1.5, alpha=0.8)
            
            if mode == 'clean' and len(lines) > 0:
                ax.add_collection(LineCollection(lines, colors=colors, linewidths=2.5))
                ax.text(0.98, 0.98, '0°→E, 90°→N', 
                        transform=ax.transAxes, 
                        ha='right', va='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return self._render_and_save('obb_directionality', plot_obb)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def visualize_all_metrics(
    analyzer,
    output_dir: str = "figures",
    figure_dpi: int = 300
) -> Dict[str, Dict[str, str]]:
    """
    Generate all four metric visualizations in a single convenience call.
    
    This function creates a SpatialMetricsVisualizer instance and executes all available
    metric visualization methods sequentially. Each metric generates three figures
    (clean, overlay, combined).
    
    Metrics visualized:
    1. Nearest Neighbor Distance: Jamming risk identification
    2. Passability Index: Vehicle navigability corridors
    3. Solidity Rugosity: Polygon shape complexity (exemplar grid)
    4. OBB Directionality: Paleo-current orientation patterns
    
    Args:
        analyzer: SpatialMetricsAnalyzer instance with computed metrics
        output_dir: Directory to save all figures (default: 'figures/')
        figure_dpi: Resolution in dots per inch (default: 300 for publication quality)
    
    Returns:
        Dict[str, Dict[str, str]]: Nested mapping:
            {'nearest_neighbor_distance': {'clean': ..., 'overlay': ..., 'combined': ...},
             'passability_index': {...},
             'solidity_rugosity': {...},
             'obb_directionality': {...}}
    
    Example:
        >>> analyzer = SpatialMetricsAnalyzer(
        ...     image_path="orthomosaic.tif",
        ...     mask_path="mask.tif",
        ...     geojson_path="annotations.geojson"
        ... )
        >>> all_results = visualize_all_metrics(analyzer, output_dir="figs_output/")
        >>> nnd_combined = all_results['nearest_neighbor_distance']['combined']
        >>> print(f"NND combined figure: {nnd_combined}")
    """
    """
    Generate all metric visualizations in one call.
    
    Convenience function that creates a SpatialMetricsVisualizer and generates
    all available metric visualizations.
    
    Args:
        analyzer: SpatialMetricsAnalyzer instance
        output_dir: Directory to save figures
        figure_dpi: DPI for output figures
    
    Returns:
        Dict[str, Dict[str, str]]: Mapping of metric names to output file paths
    
    Example:
        >>> from src.analyzer import SpatialMetricsAnalyzer
        >>> from src.visualizer import visualize_all_metrics
        >>> 
        >>> analyzer = SpatialMetricsAnalyzer(
        ...     image_path="orthomosaic.tif",
        ...     mask_path="mask.tif",
        ...     geojson_path="annotations.geojson"
        ... )
        >>> 
        >>> results = visualize_all_metrics(analyzer, output_dir="figs")
        >>> print(results['nearest_neighbor_distance']['combined'])
    """
    viz = SpatialMetricsVisualizer(analyzer, output_dir=output_dir, figure_dpi=figure_dpi)
    
    results = {}
    
    # Run all metric visualizations
    print("\n" + "="*60)
    print("GENERATING ALL METRIC VISUALIZATIONS")
    print("="*60 + "\n")
    
    try:
        results['nearest_neighbor_distance'] = viz.visualize_nearest_neighbor_distance()
    except Exception as e:
        print(f"[ERROR] NND visualization failed: {e}")
        results['nearest_neighbor_distance'] = None
    
    try:
        results['passability_index'] = viz.visualize_passability_index()
    except Exception as e:
        print(f"[ERROR] Passability visualization failed: {e}")
        results['passability_index'] = None
    
    try:
        results['solidity_rugosity'] = viz.visualize_solidity_rugosity()
    except Exception as e:
        print(f"[ERROR] Solidity visualization failed: {e}")
        results['solidity_rugosity'] = None
    
    try:
        results['obb_directionality'] = viz.visualize_obb_directionality()
    except Exception as e:
        print(f"[ERROR] OBB visualization failed: {e}")
        results['obb_directionality'] = None
    
    # Phase 3: Verticality Metrics (3D - require elevation)
    try:
        results['protrusion'] = viz.visualize_protrusion()
    except Exception as e:
        print(f"[ERROR] Protrusion visualization failed: {e}")
        results['protrusion'] = None
    
    try:
        results['embedment_angle'] = viz.visualize_embedment_angle()
    except Exception as e:
        print(f"[ERROR] Embedment Angle visualization failed: {e}")
        results['embedment_angle'] = None
    
    print("\n" + "="*60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*60)
    
    return results
