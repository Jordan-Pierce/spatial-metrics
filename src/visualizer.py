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
import matplotlib
# Force the non-interactive Agg backend before importing pyplot.
# This prevents Jupyter / IPython from ever trying to inline-display a
# partially-constructed figure when an exception is raised mid-render.
# All output goes through explicit savefig() calls; no display is needed.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
import rasterio
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy import ndimage
from scipy.spatial import KDTree
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
        # Per-metric legend specifications (populated by plotting routines).
        # Each entry: metric_name -> {'cmap': <cmap>, 'norm': <Normalize|None>, 'label': <str>, 'orientation': 'vertical'|'horizontal'}
        self._legend_specs: Dict[str, Dict[str, Any]] = {}
        # Storage for inset (mini-plot) specifications to save separately
        self._inset_specs: Dict[str, Dict[str, Any]] = {}
        
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
                raw = src.read(1).astype(np.float32)
                self.elevation_transform = src.transform

                # Handle nodata values by replacing with NaNs
                if src.nodata is not None:
                    raw[raw == src.nodata] = np.nan

                # Apply analyzer vertical scale (z_scale) so visuals match analyzer units
                self.elevation_array = raw * float(getattr(self.analyzer, 'z_scale', 1.0))
            
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
    
    def _prepare_axis_overlay_downsampled(self, ax: plt.Axes, max_px: int = 512) -> plt.Axes:
        """
        Prepare axis with a downsampled orthomosaic background for low-res figures.

        ``_prepare_axis_overlay`` passes the full-resolution orthomosaic array
        directly to ``ax.imshow``.  Matplotlib's Agg renderer must allocate a
        pixel buffer at least as large as the source array *before* it can
        rescale it to the figure canvas, so even a 96-DPI figure will crash if
        the orthomosaic is, say, 8000×6000 px (≈ 576 MB for float32 RGB).

        This helper strides the orthomosaic down to at most ``max_px`` pixels on
        its longest edge before calling ``imshow``, keeping the Agg allocation
        well within a few MB regardless of the original image size.  The world-
        coordinate extent is preserved identically so metric geometry still
        aligns correctly.

        Args:
            ax:     Matplotlib Axes to configure.
            max_px: Maximum edge length (pixels) of the downsampled image
                    passed to imshow (default 512).

        Returns:
            plt.Axes: Configured axis with downsampled orthomosaic displayed,
                      or clean white background if orthomosaic_array is None.
        """
        if self.orthomosaic_array is None:
            return self._prepare_axis_clean(ax)

        height, width = self.orthomosaic_array.shape[:2]
        stride = max(1, int(np.ceil(max(height, width) / max_px)))
        img = self.orthomosaic_array[::stride, ::stride]

        max_x = width * self.analyzer.meters_per_pixel
        max_y = height * self.analyzer.meters_per_pixel
        extent = [0, max_x, max_y, 0]

        if img.ndim == 2:
            ax.imshow(img, cmap='gray', origin='upper', extent=extent)
        else:
            ax.imshow(img, origin='upper', extent=extent)

        ax.set_xlim(0, max_x)
        ax.set_ylim(max_y, 0)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        return ax

    def _calculate_safe_stride(self, array_shape: Tuple[int, int], max_dimension: int = 2500) -> int:
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
        try:
            self._prepare_axis_clean(ax)
            plot_func(ax, 'clean')
            clean_path = self.output_dir / f"{metric_name}_clean.png"
            plt.savefig(clean_path, dpi=self.figure_dpi, bbox_inches='tight')
            output_paths['clean'] = str(clean_path)
        finally:
            plt.close(fig)

        # 2. STANDALONE OVERLAY
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.figure_dpi)
        try:
            self._prepare_axis_overlay(ax)
            plot_func(ax, 'overlay')
            overlay_path = self.output_dir / f"{metric_name}_overlay.png"
            plt.savefig(overlay_path, dpi=self.figure_dpi, bbox_inches='tight')
            output_paths['overlay'] = str(overlay_path)
        finally:
            plt.close(fig)

        # 3. COMBINED SUBPLOT (1x2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figure_size[0]*2, self.figure_size[1]), dpi=self.figure_dpi)
        try:
            self._prepare_axis_clean(ax1)
            plot_func(ax1, 'clean')
            ax1.set_title(f'{metric_name} (Clean)', fontsize=14, fontweight='bold')
            self._prepare_axis_overlay(ax2)
            plot_func(ax2, 'overlay')
            ax2.set_title(f'{metric_name} (Overlay)', fontsize=14, fontweight='bold')
            combined_path = self.output_dir / f"{metric_name}_combined.png"
            plt.savefig(combined_path, dpi=self.figure_dpi, bbox_inches='tight')
            output_paths['combined'] = str(combined_path)
        finally:
            plt.close(fig)

        return output_paths

    def _render_and_save_low_res(
        self,
        metric_name: str,
        plot_func: Callable[[plt.Axes, str], None],
        dpi: int = 96,
        figsize: Tuple[int, int] = (6, 5),
    ) -> Dict[str, str]:
        """
        Low-memory rendering engine for coarse grid-cell overview figures.

        Identical contract to _render_and_save but uses a small DPI and compact
        figure size so figures that only need to show low-resolution grid cells
        (quadrat choropleth, KDE heatmap) never trigger memory allocation failures.

        The default dpi=96 at figsize=(6,5) produces a ~576×480 px canvas — large
        enough to read cell annotations but orders of magnitude smaller than the
        ~7200×3000 px canvas that _render_and_save would create at 300 DPI.

        Args:
            metric_name: Base name for output filenames.
            plot_func:   Callback with signature plot_func(ax, mode: str).
            dpi:         Raster resolution (default 96).
            figsize:     Figure dimensions in inches (default (6, 5)).

        Returns:
            Dict[str, str]: {'clean': ..., 'overlay': ..., 'combined': ...}
        """
        output_paths = {}

        # 1. STANDALONE CLEAN
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        try:
            self._prepare_axis_clean(ax)
            plot_func(ax, 'clean')
            clean_path = self.output_dir / f"{metric_name}_clean.png"
            plt.savefig(clean_path, dpi=dpi, bbox_inches='tight')
            output_paths['clean'] = str(clean_path)
        finally:
            plt.close(fig)

        # 2. STANDALONE OVERLAY — downsampled background to avoid Agg buffer OOM
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        try:
            self._prepare_axis_overlay_downsampled(ax)
            plot_func(ax, 'overlay')
            overlay_path = self.output_dir / f"{metric_name}_overlay.png"
            plt.savefig(overlay_path, dpi=dpi, bbox_inches='tight')
            output_paths['overlay'] = str(overlay_path)
        finally:
            plt.close(fig)

        # 3. COMBINED SUBPLOT (1x2) — keep combined figure compact too
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
        try:
            self._prepare_axis_clean(ax1)
            plot_func(ax1, 'clean')
            ax1.set_title(f'{metric_name} (Clean)', fontsize=11, fontweight='bold')
            self._prepare_axis_overlay_downsampled(ax2)
            plot_func(ax2, 'overlay')
            ax2.set_title(f'{metric_name} (Overlay)', fontsize=11, fontweight='bold')
            combined_path = self.output_dir / f"{metric_name}_combined.png"
            plt.savefig(combined_path, dpi=dpi, bbox_inches='tight')
            output_paths['combined'] = str(combined_path)
        finally:
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
        # Prefer analyzer-provided cached nearest-neighbour results (default method='edge')
        nnd_cache = self.analyzer.get_nearest_neighbor_data(method='edge', recompute=False)
        metric_results = nnd_cache.get('metrics') if isinstance(nnd_cache.get('metrics'), dict) else self.analyzer.calculate_nearest_neighbor_distance(method='edge')

        polygons_all = self.analyzer._load_polygons()

        # Attempt to use cached arrays (nnd values in meters and nn indices). If cache is missing
        # or lengths mismatch, fall back to centroid KDTree proxy for visualization.
        nnd_values_m = np.asarray(nnd_cache.get('nnd_values_m', []))
        nn_indices = np.asarray(nnd_cache.get('nn_indices', []))

        if nnd_values_m.size != len(polygons_all) or nn_indices.size != len(polygons_all):
            # Fallback: compute centroid KDTree proxy
            centroids = np.array([[p.centroid.x * self.analyzer.meters_per_pixel,
                                   p.centroid.y * self.analyzer.meters_per_pixel]
                                  for p in polygons_all])
            dists_query, idxs_nn = KDTree(centroids).query(centroids, k=2)
            nn_indices = idxs_nn[:, 1]
            nnd_values_m = dists_query[:, 1]

        vmin_nnd = np.percentile(nnd_values_m, 5)
        vmax_nnd = np.percentile(nnd_values_m, 95)
        nnd_cmap = plt.cm.RdYlGn   # Red=tight, Green=spacious
        nnd_norm = Normalize(vmin=vmin_nnd, vmax=vmax_nnd)
        
        # Register legend spec instead of creating an inline colorbar.
        # Legend images will be saved later by _save_legends().
        self._legend_specs['nearest_neighbor_distance'] = {
            'cmap': nnd_cmap,
            'norm': nnd_norm,
            'label': 'Gap to nearest neighbour (m)',
            'orientation': 'vertical'
        }
        def plot_nnd(ax: plt.Axes, mode: str) -> None:
            """Render NND: polygons coloured by gap score + connector lines + inset histogram."""
            polygons = self.analyzer._load_polygons()

            # --- Draw polygons colour-coded by their NND ---
            for i, poly in enumerate(polygons):
                coords = np.array(poly.exterior.coords) * self.analyzer.meters_per_pixel
                nnd_val = nnd_values_m[i] if i < len(nnd_values_m) else vmax_nnd
                face_color = nnd_cmap(nnd_norm(nnd_val))
                if mode == 'clean':
                    patch = mpatches.Polygon(coords, fill=True,
                                             facecolor=(*face_color[:3], 0.75),
                                             edgecolor=(*face_color[:3], 1.0),
                                             linewidth=0.6)
                else:
                    patch = mpatches.Polygon(coords, fill=True,
                                             facecolor=(*face_color[:3], 0.55),
                                             edgecolor=(*face_color[:3], 0.9),
                                             linewidth=0.8)
                ax.add_patch(patch)

            # --- Draw connector lines to nearest neighbour, coloured by gap ---
            line_segs, line_colors = [], []
            for i, poly in enumerate(polygons):
                j = nn_indices[i]
                if j >= len(polygons):
                    continue
                nnd_val = nnd_values_m[i] if i < len(nnd_values_m) else vmax_nnd
                c = nnd_cmap(nnd_norm(nnd_val))
                cx_i = poly.centroid.x * self.analyzer.meters_per_pixel
                cy_i = poly.centroid.y * self.analyzer.meters_per_pixel
                cx_j = polygons[j].centroid.x * self.analyzer.meters_per_pixel
                cy_j = polygons[j].centroid.y * self.analyzer.meters_per_pixel
                line_segs.append([[cx_i, cy_i], [cx_j, cy_j]])
                line_colors.append(c)
            lc = LineCollection(line_segs, colors=line_colors,
                                linewidths=0.8, alpha=0.45, zorder=3)
            ax.add_collection(lc)

            # Colourbar removed from inline figures; legend spec recorded for later saving
            pass

            # --- Inset histogram of NND distribution ---
            # Instead of drawing the inset inline, record spec for later saving as a
            # standalone inset image to avoid cluttering each plot with small legends.
            if len(nnd_values_m) > 1:
                self._inset_specs['nearest_neighbor_distance'] = {
                    'type': 'hist',
                    'values': nnd_values_m.copy(),
                    'threshold': distance_threshold_m,
                    'cmap': nnd_cmap,
                    'norm': nnd_norm
                }

        return self._render_and_save('nearest_neighbor_distance', plot_nnd)

    def visualize_passability_index(self) -> Dict[str, str]:
        """
        Visualize passability index (navigable corridors for collection vehicles).

        Generates three figures showing where collection vehicles can navigate without
        colliding with objects. Displays a 2D distance transform heatmap (distance to
        nearest obstacle).

        Figures:
        - Clean: Distance heatmap (colorbar saved separately)
        - Overlay: Same heatmap over orthomosaic (semi-transparent)
        - Combined: 1x2 subplot for comparison
        
        Returns:
            Dict[str, str]: Output file paths {'clean': ..., 'overlay': ..., 'combined': ...}
        """
        metric_results = self.analyzer.calculate_passability_index()
        
        def plot_passability(ax: plt.Axes, mode: str) -> None:
            """Render passability as traversability corridors with contour bands."""
            mask = self.analyzer._load_mask()
            
            # Substrate (1) and Nodules (2) are free space for navigation
            # Organisms (3) and Obstructions (4) are obstacles
            free_space_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
            
            distance_transform = ndimage.distance_transform_edt(free_space_mask)
            distance_transform_m = distance_transform * self.analyzer.meters_per_pixel

            height, width = mask.shape
            extent = [0, width * self.analyzer.meters_per_pixel,
                      height * self.analyzer.meters_per_pixel, 0]

            # --- Background heatmap ---
            alpha = 1.0 if mode == 'clean' else 0.50
            im = ax.imshow(distance_transform_m, cmap='magma_r',
                           origin='upper', extent=extent, alpha=alpha)

            # --- Traversability contour bands ---
            # Build a pixel-space meshgrid for contour plotting
            h, w = distance_transform_m.shape
            xs = np.linspace(0, w * self.analyzer.meters_per_pixel, w)
            ys = np.linspace(0, h * self.analyzer.meters_per_pixel, h)
            max_r = float(distance_transform_m[~np.isnan(distance_transform_m)].max())
            
            # Contour levels at 25 %, 50 %, 75 % of max radius = vehicle size thresholds
            contour_levels = [max_r * f for f in (0.1, 0.25, 0.50) if max_r * f > 0]
            if contour_levels:
                cs = ax.contour(xs, ys, distance_transform_m,
                                levels=contour_levels, origin='upper',
                                colors=['#FF4B4B', '#FFD33D', '#3DDC84'],
                                linewidths=[0.9, 1.1, 1.3], alpha=0.85)
                contour_labels = [f'{v:.2f} m' for v in contour_levels]
                fmt = {lvl: lbl for lvl, lbl in zip(cs.levels, contour_labels)}
                ax.clabel(cs, fmt=fmt, fontsize=7, colors='white', inline=True, inline_spacing=4)

                # (Maximum inscribed circle visualization removed)

            # --- Polygon overlays: draw annotated polygons on top of the heatmap ---
            try:
                # Prefer loading polygons with class labels when available
                polygons_by_class = None
                try:
                    polygons_by_class = self.analyzer._load_polygons_with_classes()
                except Exception:
                    polygons_by_class = None

                if polygons_by_class:
                    # Draw polygons class-aware
                    for cls_name, poly_list in polygons_by_class.items():
                        cls_lower = (cls_name or '').lower()
                        is_bad = ('organ' in cls_lower) or ('obstruct' in cls_lower) or ('obstr' in cls_lower)
                        for poly in poly_list:
                            try:
                                coords_m = np.array(poly.exterior.coords) * self.analyzer.meters_per_pixel
                            except Exception:
                                continue
                            edge_col = '#FF4B4B' if is_bad else '#3DDC84'
                            lw = 0.9 if is_bad else 0.7
                            alpha_poly = 0.95 if mode == 'clean' else 0.9
                            patch = mpatches.Polygon(coords_m, fill=False,
                                                     edgecolor=edge_col, linewidth=lw,
                                                     alpha=alpha_poly, zorder=6)
                            ax.add_patch(patch)
                else:
                    # Fallback: draw unlabelled polygons in green
                    polygons = self.analyzer._load_polygons()
                    for poly in polygons:
                        try:
                            coords_m = np.array(poly.exterior.coords) * self.analyzer.meters_per_pixel
                        except Exception:
                            continue
                        edge_col = '#3DDC84'
                        lw = 0.7
                        alpha_poly = 0.9
                        patch = mpatches.Polygon(coords_m, fill=False,
                                                 edgecolor=edge_col, linewidth=lw,
                                                 alpha=alpha_poly, zorder=6)
                        ax.add_patch(patch)
            except Exception:
                # Non-fatal: polygon plotting is best-effort
                pass

            # Colourbar/legend should not be drawn inline — record a legend spec
            # so a separate legend image can be saved later via _save_legends().
            try:
                self._legend_specs['passability_index'] = {
                    'cmap': plt.cm.magma_r,
                    'norm': Normalize(vmin=0.0, vmax=max_r if max_r > 0 else 1.0),
                    'label': 'Clearance to nearest obstacle (m)',
                    'orientation': 'vertical'
                }
            except Exception:
                # Best-effort; do not break rendering if normalization fails
                pass

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
            Draw a single solidity exemplar cell showing the polygon, convex hull, and gap area.
            
            Gap area (hull minus polygon) is filled in a distinct colour to make the
            'missing area' that defines solidity immediately visible.
            """
            # Get polygon bounds in pixel space (original raster coordinates)
            px_min, py_min, px_max, py_max = poly.bounds
            width_px = px_max - px_min
            height_px = py_max - py_min
            pad_px = width_px * padding_fraction
            pad_py = height_px * padding_fraction
            crop_px_min = int(px_min - pad_px)
            crop_py_min = int(py_min - pad_py)
            crop_px_max = int(px_max + pad_px)
            crop_py_max = int(py_max + pad_py)
            
            if self.orthomosaic_array is not None:
                img_height, img_width = self.orthomosaic_array.shape[:2]
                crop_px_min = max(0, crop_px_min)
                crop_py_min = max(0, crop_py_min)
                crop_px_max = min(img_width, crop_px_max)
                crop_py_max = min(img_height, crop_py_max)
            
            crop_world_xmin = crop_px_min * self.analyzer.meters_per_pixel
            crop_world_ymin = crop_py_min * self.analyzer.meters_per_pixel
            crop_world_xmax = crop_px_max * self.analyzer.meters_per_pixel
            crop_world_ymax = crop_py_max * self.analyzer.meters_per_pixel
            
            if mode == 'clean':
                ax.set_facecolor('#161B22')
            elif mode == 'overlay' and self.orthomosaic_array is not None:
                img_crop = self.orthomosaic_array[crop_py_min:crop_py_max, crop_px_min:crop_px_max]
                if img_crop.size > 0:
                    extent = [crop_world_xmin, crop_world_xmax, crop_world_ymax, crop_world_ymin]
                    ax.imshow(img_crop, origin='upper', extent=extent)
            
            poly_coords_m = np.array(poly.exterior.coords) * self.analyzer.meters_per_pixel

            # --- Draw gap area (hull minus polygon interior) first so polygon sits on top ---
            hull = poly.convex_hull
            if hull.geom_type == 'Polygon':
                hull_coords_m = np.array(hull.exterior.coords) * self.analyzer.meters_per_pixel
                # Fill the gap area between hull and polygon with a distinct colour
                try:
                    from shapely.geometry import Polygon as ShapelyPolygon
                    gap_poly = hull.difference(poly)
                    if not gap_poly.is_empty:
                        from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
                        gap_geoms = (list(gap_poly.geoms)
                                     if gap_poly.geom_type == 'MultiPolygon'
                                     else [gap_poly])
                        for gp in gap_geoms:
                            if gp.geom_type == 'Polygon':
                                gp_coords = np.array(gp.exterior.coords) * self.analyzer.meters_per_pixel
                                ax.add_patch(mpatches.Polygon(
                                    gp_coords, fill=True,
                                    facecolor=(1.0, 0.29, 0.29, 0.45),   # translucent red = missing area
                                    edgecolor='none'))
                except Exception:
                    pass  # Gap computation is best-effort

                # Draw polygon body
                ax.add_patch(mpatches.Polygon(
                    poly_coords_m, fill=True,
                    facecolor=(0.22, 0.82, 0.85, 0.60),
                    edgecolor=(0.22, 0.82, 0.85, 1.0), linewidth=0.8))

                # Draw hull outline
                ax.add_patch(mpatches.Polygon(
                    hull_coords_m, fill=False,
                    edgecolor='#3DDC84', linewidth=1.8, linestyle='--', alpha=0.9))

                # --- Solidity gauge bar at bottom of cell ---
                bar_y = crop_world_ymax - (crop_world_ymax - crop_world_ymin) * 0.06
                bar_x0 = crop_world_xmin + (crop_world_xmax - crop_world_xmin) * 0.05
                bar_x1 = crop_world_xmin + (crop_world_xmax - crop_world_xmin) * 0.95
                bar_w = bar_x1 - bar_x0
                bar_h = (crop_world_ymax - crop_world_ymin) * 0.025
                # Background track
                ax.add_patch(mpatches.FancyBboxPatch(
                    (bar_x0, bar_y), bar_w, bar_h,
                    boxstyle='round,pad=0', facecolor='#21262D', edgecolor='#30363D',
                    linewidth=0.5, zorder=6))
                # Fill proportional to solidity
                fill_color = plt.cm.RdYlGn(solidity)
                ax.add_patch(mpatches.FancyBboxPatch(
                    (bar_x0, bar_y), bar_w * solidity, bar_h,
                    boxstyle='round,pad=0', facecolor=fill_color, edgecolor='none', zorder=7))

            ax.set_xlim(crop_world_xmin, crop_world_xmax)
            ax.set_ylim(crop_world_ymax, crop_world_ymin)
            ax.set_aspect('equal')
            # Title: coloured by solidity value
            title_color = plt.cm.RdYlGn(solidity)
            ax.set_title(f'Solidity  {solidity:.3f}', fontsize=11, fontweight='semibold',
                         color=title_color, pad=5)
            ax.axis('off')

        output_paths = {}
        
        # 1. Standalone Clean Grid
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.figure_dpi)
        try:
            for ax, idx in zip(axes.flatten(), self._exemplar_indices):
                draw_cell(ax, polygons[idx], morphology_df.iloc[idx]['solidity'], 'clean')
            plt.tight_layout()
            clean_path = self.output_dir / "solidity_rugosity_clean.png"
            plt.savefig(clean_path, dpi=self.figure_dpi)
            output_paths['clean'] = str(clean_path)
        finally:
            plt.close(fig)

        # 2. Standalone Overlay Grid
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.figure_dpi)
        try:
            for ax, idx in zip(axes.flatten(), self._exemplar_indices):
                draw_cell(ax, polygons[idx], morphology_df.iloc[idx]['solidity'], 'overlay')
            plt.tight_layout()
            overlay_path = self.output_dir / "solidity_rugosity_overlay.png"
            plt.savefig(overlay_path, dpi=self.figure_dpi)
            output_paths['overlay'] = str(overlay_path)
        finally:
            plt.close(fig)

        # 3. Combined Grid (Rows = Exemplars, Col1 = Clean, Col2 = Overlay)
        fig, axes = plt.subplots(len(self._exemplar_indices), 
                                 2, 
                                 figsize=(10, 4 * len(self._exemplar_indices)), 
                                 dpi=self.figure_dpi)
        try:
            for row_idx, idx in enumerate(self._exemplar_indices):
                draw_cell(axes[row_idx, 0], polygons[idx], morphology_df.iloc[idx]['solidity'], 'clean')
                draw_cell(axes[row_idx, 1], polygons[idx], morphology_df.iloc[idx]['solidity'], 'overlay')
            plt.tight_layout()
            combined_path = self.output_dir / "solidity_rugosity_combined.png"
            plt.savefig(combined_path, dpi=self.figure_dpi)
            output_paths['combined'] = str(combined_path)
        finally:
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
            # Record legend spec (do not create inline colorbar). Use sensible vmin/vmax.
            try:
                vmax_val = float(np.nanmax(stick_up_masked))
            except Exception:
                vmax_val = 0.0
            self._legend_specs['protrusion'] = {
                'cmap': plt.cm.plasma,
                'norm': Normalize(vmin=0, vmax=vmax_val),
                'label': 'Stick-up Height (meters)',
                'orientation': 'vertical'
            }
            
            ax.set_title('Protrusion: Collector Clearance Map', fontsize=12, fontweight='bold')
        
        # --- 3D PLOTTING LOGIC ---
        def plot_protrusion_3d(ax: plt.Axes, mode: str) -> None:
            """
            Render 3D protrusion with stick-up-coloured terrain and wireframe seafloor plane.
            
            Clean mode: Terrain coloured by stick-up height (plasma: dark=flush, bright=tall).
            Overlay mode: RGB texture draped over terrain with nodule peaks marked.
            Both modes include a sparse wireframe seafloor grid so protruding objects
            visually break through the plane.
            """
            elev_dec = elevation[::stride, ::stride]
            x = np.linspace(0, mask.shape[1] * meters_per_px, elev_dec.shape[1])
            y = np.linspace(0, mask.shape[0] * meters_per_px, elev_dec.shape[0])
            X, Y = np.meshgrid(x, y)

            # 1. Wireframe seafloor plane (sparser grid = cleaner glass-floor look)
            wf_step = max(1, elev_dec.shape[0] // 30)
            Z_floor = np.full_like(elev_dec, seafloor_z)
            ax.plot_wireframe(X[::wf_step, ::wf_step],
                              Y[::wf_step, ::wf_step],
                              Z_floor[::wf_step, ::wf_step],
                              color='#39D0D8', linewidth=0.3, alpha=0.35, zorder=1)

            # 2. Terrain surface
            if mode == 'clean':
                # Colour by stick-up above seafloor — the actual metric
                stick_up_dec = np.clip(elev_dec - seafloor_z, 0, None)
                su_max = float(np.nanmax(stick_up_dec)) or 1.0
                norm_su = Normalize(vmin=0, vmax=su_max)
                face_colors = plt.cm.plasma(norm_su(stick_up_dec))
                ax.plot_surface(X, Y, elev_dec,
                                facecolors=face_colors,
                                linewidth=0, antialiased=True, alpha=0.92, zorder=2)
                # Record legend spec for protrusion 3D (shares cmap/norm with 2D)
                self._legend_specs['protrusion'] = {
                    'cmap': plt.cm.plasma,
                    'norm': norm_su,
                    'label': 'Stick-up (m)',
                    'orientation': 'vertical'
                }

            elif mode == 'overlay' and self.orthomosaic_array is not None:
                img_dec = self.orthomosaic_array[::stride, ::stride]
                ax.plot_surface(X, Y, elev_dec,
                                facecolors=img_dec,
                                linewidth=0, antialiased=True, shade=False,
                                alpha=0.88, zorder=2)

            # 3. Scatter nodule peaks above seafloor plane
            nodule_mask_dec = (mask[::stride, ::stride] == 2)
            if np.any(nodule_mask_dec):
                elev_nodules = elev_dec[nodule_mask_dec]
                above = elev_nodules > (seafloor_z + 1e-4)
                if np.any(above):
                    peak_y_idx, peak_x_idx = np.where(nodule_mask_dec & (elev_dec > seafloor_z + 1e-4))
                    px_m = x[peak_x_idx]
                    py_m = y[peak_y_idx]
                    pz_m = elev_dec[peak_y_idx, peak_x_idx]
                    su_vals = np.clip(pz_m - seafloor_z, 0, None)
                    su_max_sc = float(su_vals.max()) or 1.0
                    scatter_colors = plt.cm.plasma(su_vals / su_max_sc)
                    ax.scatter(px_m, py_m, pz_m,
                               c=scatter_colors, s=6, alpha=0.7,
                               depthshade=False, zorder=5)

            ax.set_title('3D Protrusion  ·  Virtual Seafloor Plane', color='#E6EDF3')
        
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
        # Compute gradients using physical spacing (meters per pixel)
        dz_dy, dz_dx = np.gradient(elevation, meters_per_px)
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
            
            extent = [0, mask.shape[1] * meters_per_px, mask.shape[0] * meters_per_px, 0]
            
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
            # Record legend spec for embedment angle (no inline colorbar)
            self._legend_specs['embedment_angle'] = {
                'cmap': plt.cm.RdYlGn_r,
                'norm': Normalize(vmin=0, vmax=90),
                'label': 'Contact Slope / Embedment Angle (°)',
                'orientation': 'vertical'
            }
            
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
                
                # Record legend spec for embedment angle (3D surface)
                self._legend_specs['embedment_angle'] = {
                    'cmap': plt.cm.RdYlGn_r,
                    'norm': norm,
                    'label': 'Surface Gradient (°)',
                    'orientation': 'vertical'
                }
            
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
                # Record legend spec for scatter-based contact angle (3D overlay)
                self._legend_specs['embedment_angle'] = {
                    'cmap': plt.cm.RdYlGn_r,
                    'norm': Normalize(vmin=0, vmax=90),
                    'label': 'Contact Angle (°)',
                    'orientation': 'vertical'
                }
            
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
        
        # Pre-compute principal angles for all polygons
        polygons_all = self.analyzer._load_polygons()
        angles_deg = []
        for i, poly in enumerate(polygons_all):
            obb = poly.minimum_rotated_rectangle
            if obb.geom_type != 'Polygon':
                angles_deg.append(0.0)
                continue
            obb_coords = np.array(obb.exterior.coords)
            # Compute vectors along two adjacent OBB sides
            s1 = obb_coords[1] - obb_coords[0]
            s2 = obb_coords[2] - obb_coords[1]
            # Use longest side vector as principal vector
            pv = s1 if np.linalg.norm(s1) > np.linalg.norm(s2) else s2
            # Normalize and compute angle
            pv = pv / (np.linalg.norm(pv) + 1e-9)
            angle_rad = np.arctan2(pv[1], pv[0])
            angles_deg.append(np.degrees(angle_rad) % 360)
        angles_deg = np.array(angles_deg)

        # Legend spec for orientation (cyclic HSV colormap, degrees 0-360)
        self._legend_specs['obb_directionality'] = {
            'cmap': plt.cm.hsv,
            'norm': Normalize(vmin=0, vmax=360),
            'label': 'OBB Orientation (°)',
            'orientation': 'vertical'
        }

        def plot_obb(ax: plt.Axes, mode: str) -> None:
            """Render OBB: polygons + directional lines coloured by orientation."""
            polygons = self.analyzer._load_polygons()
            lines, line_colors = [], []

            if mode == 'clean':
                # Add semi-transparent gray polygon backgrounds
                for poly in polygons:
                    coords = (
                        np.array(poly.exterior.coords) 
                        * self.analyzer.meters_per_pixel
                    )
                    ax.add_patch(
                        mpatches.Polygon(
                            coords, 
                            fill=True,
                            facecolor=(0.22, 0.82, 0.85, 0.18),
                            edgecolor=(0.22, 0.82, 0.85, 0.45),
                            linewidth=0.4
                        )
                    )

            # Render directional indicators or OBB rectangles
            for i, poly in enumerate(polygons):
                if i >= len(morphology_df) or i >= len(angles_deg):
                    continue
                
                angle = angles_deg[i]
                color = hsv_to_rgb([angle / 360.0, 0.85, 0.95])
                cx = morphology_df.iloc[i]['centroid_x_m']
                cy = morphology_df.iloc[i]['centroid_y_m']
                rad = np.radians(angle)
                pv = np.array([np.cos(rad), np.sin(rad)])

                if mode == 'clean':
                    # Draw directional lines with length = OBB width (shorter dimension)
                    try:
                        obb = poly.minimum_rotated_rectangle
                        if obb.geom_type == 'Polygon':
                            obb_coords = np.array(obb.exterior.coords)
                            s1 = obb_coords[1] - obb_coords[0]
                            s2 = obb_coords[2] - obb_coords[1]
                            side1 = (
                                np.linalg.norm(s1) 
                                * self.analyzer.meters_per_pixel
                            )
                            side2 = (
                                np.linalg.norm(s2) 
                                * self.analyzer.meters_per_pixel
                            )
                            # Use shorter dimension (width) for line length
                            half_len = min(side1, side2) / 2.0
                        else:
                            raise ValueError("degenerate obb")
                    except Exception:
                        # Fallback to polygon bounding box if OBB fails
                        bounds = poly.bounds
                        extent_x = bounds[2] - bounds[0]
                        extent_y = bounds[3] - bounds[1]
                        min_extent = (
                            min(extent_x, extent_y) 
                            * self.analyzer.meters_per_pixel
                        )
                        half_len = min_extent / 2.0

                    start = np.array([cx, cy]) - pv * half_len
                    end = np.array([cx, cy]) + pv * half_len
                    lines.append([start, end])
                    line_colors.append(color)
                else:
                    # Overlay mode: draw OBB rectangles colored by angle
                    obb = poly.minimum_rotated_rectangle
                    if obb.geom_type == 'Polygon':
                        obb_coords_m = (
                            np.array(obb.exterior.coords) 
                            * self.analyzer.meters_per_pixel
                        )
                        ax.add_patch(
                            mpatches.Polygon(
                                obb_coords_m,
                                fill=True,
                                facecolor=(*color, 0.28),
                                edgecolor=(*color, 0.85),
                                linewidth=1.2
                            )
                        )

            # Add rendered directional lines (clean mode only)
            if mode == 'clean' and lines:
                ax.add_collection(
                    LineCollection(
                        lines,
                        colors=line_colors,
                        linewidths=1.5,
                        alpha=0.88,
                        zorder=4
                    )
                )

            # Record rose diagram inset for separate rendering
            if len(angles_deg) > 0:
                self._inset_specs['obb_directionality'] = {
                    'type': 'rose',
                    'angles_deg': angles_deg.copy(),
                    'n_bins': 24
                }

            # Compass orientation label
            ax.text(
                0.99, 0.01,
                '0°→E  ·  90°→N',
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=8,
                color='#8B949E',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    fc='#0D1117',
                    ec='#30363D',
                    alpha=0.8
                )
            )

        return self._render_and_save('obb_directionality', plot_obb)

    # =========================================================================
    # RESOURCE DISTRIBUTION SUITE
    # =========================================================================

    def visualize_spatial_homogeneity(self, grid_size: int = 16, target_class: int = 2) -> Dict[str, str]:
        """
        Visualize Spatial Homogeneity using the Semantic Mask, overlaid with polygons.
        
        Divides the mask into a grid and calculates the fractional area coverage
        of the target class. Renders as a clean heatmap with actual polygon 
        outlines drawn on top for precise location context.
        """
        print(f"\n[VIS] Generating Spatial Homogeneity & Polygons (Grid: {grid_size}x{grid_size}, Class: {target_class})...")
        
        mask = self.analyzer._load_mask()
        if mask is None:
            return {}
            
        # 1. Create a binary mask of just our target class
        binary_mask = (mask == target_class).astype(float)
        
        # 2. Chop into grid cells and calculate coverage percentage
        h, w = binary_mask.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        # Crop to perfectly divisible size
        cropped_mask = binary_mask[:cell_h * grid_size, :cell_w * grid_size]
        
        # Reshape and take the mean to get the coverage fraction per cell
        grid_coverage = cropped_mask.reshape(grid_size, cell_h, grid_size, cell_w).mean(axis=(1, 3))
        grid_percentage = grid_coverage * 100.0
        
        # Calculate VMR on the coverage percentages
        mean_cov = np.mean(grid_percentage)
        variance = np.var(grid_percentage, ddof=1) if grid_percentage.size > 1 else 0
        vmr = variance / mean_cov if mean_cov > 0 else 0
        
        if mean_cov == 0: pattern = 'EMPTY'
        elif vmr < 0.8: pattern = 'UNIFORM'
        elif vmr > 1.2: pattern = 'CLUSTERED'
        else: pattern = 'RANDOM'
        
        # 3. Setup rendering
        extent = [0, w * self.analyzer.meters_per_pixel, h * self.analyzer.meters_per_pixel, 0]
        cmap = plt.cm.magma
        norm = Normalize(vmin=0, vmax=max(grid_percentage.max(), 1.0))
        
        self._legend_specs['spatial_homogeneity'] = {
            'cmap': cmap, 'norm': norm,
            'label': f'Class {target_class} Coverage (%)',
            'orientation': 'vertical'
        }
        
        # Load polygons for the overlay
        polygons = self.analyzer._load_polygons()
        meters_per_px = self.analyzer.meters_per_pixel
        
        def plot_homogeneity(ax: plt.Axes, mode: str) -> None:
            alpha_hm = 0.65 if mode == 'overlay' else 0.95
            
            # Plot the clean heatmap background
            ax.imshow(grid_percentage, cmap=cmap, norm=norm, origin='upper',
                      extent=extent, aspect='equal', alpha=alpha_hm, interpolation='nearest', zorder=2)
            
            # Draw all polygon outlines on top
            for poly in polygons:
                try:
                    coords_m = np.array(poly.exterior.coords) * meters_per_px
                    # Thin, semi-transparent white outline so it doesn't overpower the heatmap
                    patch = mpatches.Polygon(coords_m, fill=False, edgecolor='white', 
                                             linewidth=0.6, alpha=0.7, zorder=3)
                    ax.add_patch(patch)
                except Exception:
                    continue
                
            ax.set_title(f"Class {target_class} Coverage Homogeneity & Polygons\nVMR = {vmr:.2f} [{pattern}]", 
                         fontsize=11, fontweight='bold')
                         
        # Use low-res pipeline to prevent RAM crashes on the overlay
        return self._render_and_save_low_res('spatial_homogeneity', plot_homogeneity)

    def visualize_bivariate_ripleys_k(self, class_a: str = 'nodule', class_b: str = 'organism',
                                      halo_radii_m: Tuple[float, float, float] = (0.25, 0.50, 1.0)) -> Dict[str, str]:
        """
        Visualize the Bivariate Ripley's K (Invisible Halo).

        Three fixed halo rings (default 0.5 m, 1.5 m, 3.0 m) radiate outward from
        each biological centroid.  Nodule bounding boxes are colour-graded from
        red → yellow → green → blue based on their distance to the nearest
        biological, so proximity risk is immediately readable at a glance.

        Visual language:
          - Halos: three concentric rings at halo_radii_m, brightest/thickest at the
            innermost boundary, fading outward.  A faint filled disc fills the core.
          - Nodules: axis-aligned bounding-box rectangle, colour mapped through a
            red→yellow→green→blue gradient keyed to distance-to-nearest-biological.
            The outermost halo radius defines the "at-risk" boundary; beyond 2× that
            radius nodules are rendered in full safe-blue.
          - Biologicals: bright-green filled bounding-box square, drawn on top.

        Args:
            class_a:       Resource class label (default 'nodule').
            class_b:       Biological class label (default 'organism').
            halo_radii_m:  Three halo radii in metres, inner to outer
                           (default (0.5, 1.5, 3.0)).
        """
        print("\n[VIS] Generating Bivariate Ripley's K visualization (Invisible Halo)...")

        results = self.analyzer.calculate_bivariate_ripleys_k(class_a=class_a, class_b=class_b)
        radii   = np.asarray(results.get('radii_m', []))
        observed = np.asarray(results.get('observed_counts', []))
        expected = np.asarray(results.get('expected_poisson_counts', []))

        if radii.size == 0 or observed.size == 0:
            print("[WARNING] No Ripley's K results available; skipping bivariate viz.")
            return {}

        # Peak radius from Ripley's K curve (informational only — halos use fixed radii)
        diff = observed - expected
        idx_peak  = int(np.nanargmax(diff)) if not np.all(np.isnan(diff)) else int(np.argmax(observed))
        r_peak_m  = float(radii[idx_peak])

        meters_per_px = float(self.analyzer.meters_per_pixel or 1.0)

        # ── Halo ring definitions (inner → outer) ────────────────────────────
        r_inner, r_mid, r_outer = sorted(float(r) for r in halo_radii_m)
        halo_rings = [
            # (radius_m, linewidth, alpha, linestyle)
            (r_inner, 2.0, 0.75, '-'),
            (r_mid,   1.4, 0.45, '--'),
            (r_outer, 1.0, 0.25, ':'),
        ]
        halo_ring_color = '#3DDC84'   # bright green

        # ── Nodule distance → colour map ─────────────────────────────────────
        # Red (distance=0) → Yellow → Green → Blue (distance ≥ 2×r_outer)
        # We build a custom 4-stop LinearSegmentedColormap.
        nodule_cmap = LinearSegmentedColormap.from_list(
            'proximity_risk',
            [(0.00, '#FF2D2D'),   # 0.0  — touching: red
             (0.25, '#FF9900'),   # 0.25 — inside inner ring: orange
             (0.50, '#FFE033'),   # 0.5  — between inner and mid: yellow
             (0.75, '#3DDC84'),   # 0.75 — between mid and outer: green
             (1.00, '#5BB8F5')],  # 1.0  — beyond outer ring: safe blue
        )
        # Normalise: 0 = touching a biological, 1 = 2× outer ring or farther
        dist_norm_max = r_outer * 2.0

        # ── Biological box style ──────────────────────────────────────────────
        bio_edge  = '#3DDC84'
        bio_fill  = (0.24, 0.86, 0.52, 0.20)

        # ── Load polygons ─────────────────────────────────────────────────────
        polys_by_class = self.analyzer._load_polygons_with_classes()
        class_a_polys  = polys_by_class.get(class_a, [])
        class_b_polys  = polys_by_class.get(class_b, [])

        def _aabb_m(poly: Polygon) -> Tuple[float, float, float, float]:
            minx, miny, maxx, maxy = poly.bounds
            return (minx * meters_per_px,
                    miny * meters_per_px,
                    (maxx - minx) * meters_per_px,
                    (maxy - miny) * meters_per_px)

        # Centroids in metres
        centroids_b_m = (
            np.array([[p.centroid.x, p.centroid.y] for p in class_b_polys]) * meters_per_px
            if class_b_polys else np.zeros((0, 2))
        )
        centroids_a_m = (
            np.array([[p.centroid.x, p.centroid.y] for p in class_a_polys]) * meters_per_px
            if class_a_polys else np.zeros((0, 2))
        )

        # Distance from every nodule centroid to its nearest biological
        if centroids_b_m.shape[0] > 0 and centroids_a_m.shape[0] > 0:
            tree_b = KDTree(centroids_b_m)
            dists_a_to_b, _ = tree_b.query(centroids_a_m, k=1)  # shape (N_nodules,)
        else:
            dists_a_to_b = np.full(len(class_a_polys), np.inf)

        # Normalised distance [0, 1] for colour lookup
        norm_dists = np.clip(dists_a_to_b / dist_norm_max, 0.0, 1.0)

        # Count nodules inside each halo zone for the title
        n_inner  = int(np.sum(dists_a_to_b <= r_inner))
        n_mid    = int(np.sum((dists_a_to_b > r_inner) & (dists_a_to_b <= r_mid)))
        n_outer  = int(np.sum((dists_a_to_b > r_mid)   & (dists_a_to_b <= r_outer)))
        n_safe   = int(np.sum(dists_a_to_b > r_outer))

        print(f"  → Halo rings: {r_inner} m / {r_mid} m / {r_outer} m")
        print(f"  → Nodules — ≤{r_inner}m: {n_inner}  ≤{r_mid}m: {n_mid}  ≤{r_outer}m: {n_outer}  safe: {n_safe}")

        # Register external legend (do not draw inline colorbar in figures)
        self._legend_specs['bivariate_ripleys_k'] = {
            'cmap': nodule_cmap,
            'norm': Normalize(vmin=0, vmax=dist_norm_max),
            'label': 'Dist. to nearest biological (m)',
            'orientation': 'vertical'
        }

        def plot_bivariate(ax: plt.Axes, mode: str) -> None:
            # ── 1. Halo rings + core glow (background, drawn first) ───────
            if centroids_b_m.shape[0] > 0:
                for (bx, by) in centroids_b_m:
                    # Faint filled disc spanning the innermost ring
                    ax.add_patch(mpatches.Circle(
                        (bx, by), r_inner,
                        fill=True,
                        facecolor=(*matplotlib.colors.to_rgb(halo_ring_color), 0.08),
                        edgecolor='none',
                        zorder=1,
                    ))
                    # Three concentric rings, outer→inner so inner draws on top
                    for r_ring, lw, alpha, ls in reversed(halo_rings):
                        ax.add_patch(mpatches.Circle(
                            (bx, by), r_ring,
                            fill=False,
                            edgecolor=halo_ring_color,
                            linewidth=lw,
                            alpha=alpha,
                            linestyle=ls,
                            zorder=2,
                        ))
                    # Radius labels on clean mode (not on overlay to avoid clutter)
                    if mode == 'clean':
                        for r_ring, label in zip(
                            [r_inner, r_mid, r_outer],
                            [f'{r_inner}m', f'{r_mid}m', f'{r_outer}m'],
                        ):
                            ax.text(
                                bx + r_ring * 0.71, by - r_ring * 0.71,
                                label,
                                fontsize=6, color=halo_ring_color, alpha=0.7,
                                ha='left', va='top', zorder=5,
                            )

            # ── 2. Nodule boxes — colour-graded by distance ───────────────
            for i, poly in enumerate(class_a_polys):
                x0, y0, w, h = _aabb_m(poly)
                nd = float(norm_dists[i]) if i < len(norm_dists) else 1.0
                edge_rgba = nodule_cmap(nd)
                fill_rgba = (*edge_rgba[:3], 0.12)
                lw = 1.6 if nd < 0.5 else 0.9   # thicker border when close
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0, y0), w, h,
                    boxstyle='square,pad=0',
                    fill=True,
                    facecolor=fill_rgba,
                    edgecolor=edge_rgba,
                    linewidth=lw,
                    zorder=3,
                ))

            # ── 3. Biological boxes (top layer) ──────────────────────────
            for poly in class_b_polys:
                x0, y0, w, h = _aabb_m(poly)
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0, y0), w, h,
                    boxstyle='square,pad=0',
                    fill=True,
                    facecolor=bio_fill,
                    edgecolor=bio_edge,
                    linewidth=1.8,
                    zorder=4,
                ))

            # Colourbar/legend are intentionally not drawn inline in the figures.
            # A separate legend image is saved via the public legend saving utility
            # (see _save_legends / save_legends) to keep map figures clean.

            ax.set_title(
                f'Invisible Halo  |  rings: {r_inner} / {r_mid} / {r_outer} m  |  '
                f'≤{r_inner}m: {n_inner}  ≤{r_mid}m: {n_mid}  ≤{r_outer}m: {n_outer}  safe: {n_safe}',
                fontsize=8,
            )

        # ── Render Rule-of-Three suite ────────────────────────────────────────
        output_paths = self._render_and_save('bivariate_ripleys_k', plot_bivariate)

        # ── Ripley's K curve ──────────────────────────────────────────────────
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=self.figure_dpi)
            try:
                ax.plot(radii, observed, color='#3DDC84', linewidth=2.0, label='Observed')
                ax.plot(radii, expected, color='#FF6B6B', linestyle='--', linewidth=1.6,
                        label='Expected (Poisson)')
                ax.axvline(r_peak_m, color='#FFA500', linestyle=':', linewidth=1.4,
                           label=f'Peak r={r_peak_m:.2f} m')
                for r_ring, ls, col in zip(
                    [r_inner, r_mid, r_outer],
                    ['-', '--', ':'],
                    ['#FF2D2D', '#FFE033', '#5BB8F5'],
                ):
                    ax.axvline(r_ring, color=col, linewidth=0.9, linestyle=ls, alpha=0.6)
                ax.set_xlabel('Radius (m)')
                ax.set_ylabel('Counts')
                ax.set_xscale('log')
                ax.legend(fontsize=7)
                ax.grid(alpha=0.18)
                curve_path = self.output_dir / 'bivariate_ripleys_k_curve.png'
                plt.tight_layout()
                plt.savefig(curve_path, dpi=self.figure_dpi, bbox_inches='tight')
                output_paths['curve'] = str(curve_path)
            finally:
                plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Failed to save Ripley's K curve: {e}")

        # ── Interpretive legend ───────────────────────────────────────────────
        try:
            fig = plt.figure(figsize=(4.5, 4), dpi=self.figure_dpi)
            try:
                ax = fig.add_subplot(111)
                ax.axis('off')
                interp = results.get('interpretation', '')
                n_a    = results.get('n_class_a', 0)
                n_b    = results.get('n_class_b', 0)
                txt = (
                    f"{interp}\n\n"
                    f"{class_a.capitalize()}s: {n_a}   {class_b.capitalize()}s: {n_b}\n"
                    f"Rings: {r_inner} m / {r_mid} m / {r_outer} m\n"
                    f"Ripley K peak: {r_peak_m:.3f} m\n\n"
                    f"≤{r_inner}m: {n_inner}  |  ≤{r_mid}m: {n_mid}  |  ≤{r_outer}m: {n_outer}  |  safe: {n_safe}"
                )
                ax.text(0.02, 0.98, txt, va='top', ha='left', fontsize=8.5,
                        transform=ax.transAxes, family='monospace')

                # Gradient swatch for nodule colour scale
                grad = np.linspace(0, 1, 256).reshape(1, -1)
                ax_cb = fig.add_axes([0.05, 0.15, 0.55, 0.06])
                ax_cb.imshow(grad, aspect='auto', cmap=nodule_cmap, origin='lower')
                ax_cb.set_yticks([])
                ax_cb.set_xticks([0, 85, 170, 255])
                ax_cb.set_xticklabels(['0 m', f'{r_inner}m', f'{r_mid}m', f'{r_outer}m+'], fontsize=7)
                ax_cb.set_title('Nodule box colour = dist. to nearest biological', fontsize=7, pad=3)

                # Bio box swatch
                ax.add_patch(mpatches.Rectangle(
                    (0.05, 0.05), 0.08, 0.055,
                    facecolor=bio_fill, edgecolor=bio_edge, linewidth=1.5,
                    transform=ax.transAxes,
                ))
                ax.text(0.16, 0.075, f'{class_b.capitalize()} (biological)',
                        transform=ax.transAxes, fontsize=8, va='center')

                legend_path = self.output_dir / 'bivariate_ripleys_k_legend.png'
                plt.savefig(legend_path, dpi=self.figure_dpi, bbox_inches='tight')
                output_paths['legend'] = str(legend_path)
            finally:
                plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Failed to save bivariate legend: {e}")

        return output_paths

    def _save_legends(self) -> Dict[str, str]:
        """Iterate self._legend_specs and save a small image per metric showing only the colorbar.

        Returns a dict mapping metric_name -> file path string.
        """
        out = {}
        for metric, spec in self._legend_specs.items():
            cmap = spec.get('cmap', plt.cm.viridis)
            norm = spec.get('norm', None)
            label = spec.get('label', '')
            orientation = spec.get('orientation', 'vertical')

            # Choose figure size by orientation
            if orientation == 'horizontal':
                figsize = (4, 1)
                cbar_ax_rect = [0.12, 0.45, 0.76, 0.35]
            else:
                figsize = (2, 4)
                cbar_ax_rect = [0.35, 0.05, 0.3, 0.9]

            fig = plt.figure(figsize=figsize, dpi=self.figure_dpi)
            # Create an axes for the colorbar only
            cax = fig.add_axes(cbar_ax_rect)

            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, orientation=orientation)
            cb.set_label(label, fontsize=8)
            # Minimal tick styling
            if orientation == 'horizontal':
                cb.ax.xaxis.set_tick_params(labelsize=7)
            else:
                cb.ax.yaxis.set_tick_params(labelsize=7)

            out_path = self.output_dir / f"{metric}_legend.png"
            plt.savefig(out_path, dpi=self.figure_dpi, bbox_inches='tight')
            plt.close(fig)
            out[metric] = str(out_path)

        return out

    def save_legends(self) -> Dict[str, str]:
        """Public wrapper to generate and save legend images for recorded metrics."""
        return self._save_legends()

    def _save_insets(self) -> Dict[str, str]:
        """Save recorded inset plots (histograms, roses) as standalone images.

        Returns a dict mapping metric_name -> inset file path.
        """
        out = {}
        for metric, spec in self._inset_specs.items():
            itype = spec.get('type')

            if itype == 'hist':
                values = np.asarray(spec.get('values', []))
                threshold = spec.get('threshold', None)
                cmap = spec.get('cmap', plt.cm.viridis)
                norm = spec.get('norm', None)

                n_bins = min(30, max(10, len(values) // 5))
                fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=self.figure_dpi)
                ax.set_facecolor('#161B22')
                ax.patch.set_alpha(0.85)
                counts, bin_edges = np.histogram(values, bins=n_bins)
                # Compute bin center colors using provided cmap/norm
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                if norm is not None:
                    bin_colors = [cmap(norm(c)) for c in bin_centers]
                else:
                    # fallback linear normalization across data range
                    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
                    tmp_norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
                    bin_colors = [cmap(tmp_norm(c)) for c in bin_centers]

                ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
                       color=bin_colors, edgecolor='none', align='edge')
                if threshold is not None:
                    ax.axvline(threshold, color='#FF4B4B', linewidth=1.2, linestyle='--', alpha=0.9)
                ax.set_xlabel('Value', fontsize=8, color='#8B949E')
                ax.set_ylabel('Count', fontsize=8, color='#8B949E')
                ax.tick_params(labelsize=7, colors='#8B949E')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#30363D')

                out_path = self.output_dir / f"{metric}_inset_hist.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=self.figure_dpi, bbox_inches='tight')
                plt.close(fig)
                out[metric] = str(out_path)

            elif itype == 'rose':
                angles = np.asarray(spec.get('angles_deg', []))
                n_bins = int(spec.get('n_bins', 24))
                fig = plt.figure(figsize=(3, 3), dpi=self.figure_dpi)
                ax = fig.add_subplot(111, projection='polar')
                ax.set_facecolor('#161B22')
                ax.patch.set_alpha(0.85)
                bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
                folded = np.deg2rad(angles % 180) * 2
                counts, _ = np.histogram(folded, bins=bin_edges)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bar_width = 2 * np.pi / n_bins
                bar_colors = [hsv_to_rgb([theta / (2 * np.pi), 0.8, 0.9])
                              for theta in bin_centers]
                ax.bar(bin_centers, counts, width=bar_width, color=bar_colors,
                       edgecolor='none', alpha=0.85)
                ax.set_yticks([])
                ax.tick_params(labelsize=6, colors='#8B949E', pad=1)
                ax.set_theta_zero_location('E')
                ax.set_theta_direction(1)
                ax.spines['polar'].set_edgecolor('#30363D')
                ax.set_title('Orientation\nrose', fontsize=8, color='#8B949E', pad=2)

                out_path = self.output_dir / f"{metric}_inset_rose.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=self.figure_dpi, bbox_inches='tight')
                plt.close(fig)
                out[metric] = str(out_path)

            else:
                # Unknown inset type: skip
                continue

        return out

    def save_insets(self) -> Dict[str, str]:
        """Public wrapper to generate and save recorded inset images."""
        return self._save_insets()
    
    # ============================================================================
    # BIOLOGICAL LOSS SUITE
    # ============================================================================

    def _generate_simulated_paths(self, width_m: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """
        Generates 4 dynamic vehicle tracks. 
        Uses a Dynamic Programming algorithm to find the optimal path, augmented 
        with a "rubber band" penalty to return to the center, and Cubic Splines 
        for smooth, realistic driving curves.
        """
        from shapely.geometry import LineString
        import numpy as np
        from scipy.interpolate import CubicSpline
        
        x_min, x_max, y_min, y_max = self._get_axis_limits()
        mask = self.analyzer._load_mask()
        
        paths = {}
        x_center = (x_min + x_max) / 2.0
        
        # 1. Baseline: Straight down the middle
        center_line = LineString([(x_center, y_min - 2.0), (x_center, y_max + 2.0)])
        paths['center_cut'] = {
            # join_style=1 (round) creates smooth buffered edges around the curves
            'polygon': center_line.buffer(width_m / 2.0, cap_style=2, join_style=1),
            'centerline': center_line
        }
        
        if mask is None:
            return paths
            
        mpp = self.analyzer.meters_per_pixel
        width_px = max(1, int(width_m / mpp))
        h, w = mask.shape
        
        num_steps = 20  
        step_h = max(1, h // num_steps)
        
        stride_x = max(1, int(0.1 / mpp))  
        x_nodes = np.arange(width_px // 2, w - width_px // 2, stride_x)
        if len(x_nodes) == 0: x_nodes = np.array([w // 2])
        nx = len(x_nodes)
        
        max_steer_px = int(0.5 / mpp)
        max_steer_nodes = max(1, max_steer_px // stride_x)
        
        # Calculate the "Rubber Band" centering penalty
        # Costs the equivalent of 0.5 nodules per meter of drift from the centerline
        dist_from_center_m = np.abs(x_nodes - (w // 2)) * mpp
        pull_penalty = dist_from_center_m * 0.5
        
        dp_profit, dp_terrain, dp_eco = np.zeros(nx), np.zeros(nx), np.zeros(nx)
        hist_profit = np.zeros((num_steps, nx), dtype=int)
        hist_terrain = np.zeros((num_steps, nx), dtype=int)
        hist_eco = np.zeros((num_steps, nx), dtype=int)
        
        kernel = np.ones(width_px)
        
        for step in range(num_steps):
            y_start = step * step_h
            y_end = (step + 1) * step_h if step < num_steps - 1 else h
            
            strip_N = np.sum(mask[y_start:y_end, :] == 2, axis=0) 
            strip_H = np.sum((mask[y_start:y_end, :] == 3) | (mask[y_start:y_end, :] == 4), axis=0) 
            
            N_scores_full = np.convolve(strip_N, kernel, mode='same')
            H_scores_full = np.convolve(strip_H, kernel, mode='same')
            
            # Apply the scores MINUS the centering penalty so the vehicle wants to return to the middle
            s_profit = N_scores_full[x_nodes] - pull_penalty
            s_terrain = (N_scores_full[x_nodes] * 0.001) - (H_scores_full[x_nodes] * 1000.0) - pull_penalty
            s_eco = N_scores_full[x_nodes] - (H_scores_full[x_nodes] * 10.0) - pull_penalty
            
            if step == 0:
                dp_profit, dp_terrain, dp_eco = s_profit, s_terrain, s_eco
            else:
                new_dp_p, new_dp_t, new_dp_e = np.full(nx, -np.inf), np.full(nx, -np.inf), np.full(nx, -np.inf)
                
                for i in range(nx):
                    start_idx = max(0, i - max_steer_nodes)
                    end_idx = min(nx, i + max_steer_nodes + 1)
                    
                    best_p = np.argmax(dp_profit[start_idx:end_idx])
                    new_dp_p[i] = s_profit[i] + dp_profit[start_idx:end_idx][best_p]
                    hist_profit[step, i] = start_idx + best_p
                    
                    best_t = np.argmax(dp_terrain[start_idx:end_idx])
                    new_dp_t[i] = s_terrain[i] + dp_terrain[start_idx:end_idx][best_t]
                    hist_terrain[step, i] = start_idx + best_t
                    
                    best_e = np.argmax(dp_eco[start_idx:end_idx])
                    new_dp_e[i] = s_eco[i] + dp_eco[start_idx:end_idx][best_e]
                    hist_eco[step, i] = start_idx + best_e
                    
                dp_profit, dp_terrain, dp_eco = new_dp_p, new_dp_t, new_dp_e
                
        # Backtrack
        curr_p, curr_t, curr_e = np.argmax(dp_profit), np.argmax(dp_terrain), np.argmax(dp_eco)
        pts_p, pts_t, pts_e = [], [], []
        
        for step in range(num_steps - 1, -1, -1):
            y_start = step * step_h
            y_end = (step + 1) * step_h if step < num_steps - 1 else h
            y_mid_m = y_min + ((y_start + y_end) / 2.0) * mpp
            
            if step == 0: y_mid_m = y_min
            if step == num_steps - 1: y_mid_m = y_max
            
            pts_p.append((x_nodes[curr_p] * mpp, y_mid_m))
            pts_t.append((x_nodes[curr_t] * mpp, y_mid_m))
            pts_e.append((x_nodes[curr_e] * mpp, y_mid_m))
            
            if step > 0:
                curr_p = hist_profit[step, curr_p]
                curr_t = hist_terrain[step, curr_t]
                curr_e = hist_eco[step, curr_e]
                
        pts_p.reverse()
        pts_t.reverse()
        pts_e.reverse()
        
        strategies = {
            'profit_maximizer': pts_p,
            'terrain_aware': pts_t,
            'eco_optimized': pts_e
        }
        
        # Helper function to generate smooth curves from the zigzag waypoints
        def smooth_waypoints(pts):
            pts_arr = np.array(pts)
            x_vals = pts_arr[:, 0]
            y_vals = pts_arr[:, 1]
            
            # Fit a cubic spline (X as a function of Y, since Y is strictly increasing down the image)
            cs = CubicSpline(y_vals, x_vals)
            
            # Generate 100 high-resolution points along the curve
            y_smooth = np.linspace(y_vals[0], y_vals[-1], 100)
            x_smooth = cs(y_smooth)
            
            return list(zip(x_smooth, y_smooth))
        
        for key, pts in strategies.items():
            # 1. Extend off-screen
            extended_pts = [(pts[0][0], y_min - 2.0)] + pts + [(pts[-1][0], y_max + 2.0)]
            
            # 2. Smooth the path into a sweeping curve
            curved_pts = smooth_waypoints(extended_pts)
            
            track_line = LineString(curved_pts)
            paths[key] = {
                'polygon': track_line.buffer(width_m / 2.0, cap_style=2, join_style=1),
                'centerline': track_line
            }
            
        return paths

    def _plot_biological_loss_chart(self, direct_counts: Dict[str, int], indirect_counts: Dict[str, int], output_path: str, title: str) -> None:
        """Generates a standalone stacked bar chart for biological casualties."""
        from matplotlib.ticker import MaxNLocator
        
        # Filter out resource classes to strictly show biological loss
        bio_classes = [c for c in direct_counts.keys() if 'nodule' not in c.lower() and 'rock' not in c.lower()]
        
        if not bio_classes:
            return 
            
        classes = sorted(bio_classes)
        direct_vals = [direct_counts[c] for c in classes]
        indirect_vals = [indirect_counts[c] for c in classes]
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=self.figure_dpi)
        fig.patch.set_facecolor('#0D1117')
        ax.set_facecolor('#161B22')
        
        # Create stacked bars
        x_pos = np.arange(len(classes))
        ax.bar(x_pos, direct_vals, color='#FF4B4B', label='Direct (Killed)', edgecolor='#0D1117')
        ax.bar(x_pos, indirect_vals, bottom=direct_vals, color='#FFA500', label='Indirect (Buffer)', edgecolor='#0D1117')
        
        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.capitalize() for c in classes], color='#C9D1D9', rotation=45, ha='right')
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363D')
            
        # FORCE INTEGER Y-AXIS
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
        ax.set_ylabel('Number of Organisms Impacted', color='#C9D1D9', fontweight='bold')
        ax.set_title(title, color='#C9D1D9', fontweight='bold', pad=15)
        ax.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#C9D1D9')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close(fig)

    def visualize_projected_biological_loss(self, buffer_distance_m: float = 0.5, corridor_width_m: float = 1.0) -> Dict[str, Dict[str, str]]:
        """
        Simulates 4 different mining pathfinding strategies and generates a spatial 
        map and stacked bar chart for each, visualizing ecological trade-offs.
        """
        print(f"\n[VIS] Generating Projected Biological Loss Simulations (Vehicle: {corridor_width_m}m, Buffer: {buffer_distance_m}m)...")
        
        paths_data = self._generate_simulated_paths(width_m=corridor_width_m)
        polygons_by_class = self.analyzer._load_polygons_with_classes()
        meters_per_px = self.analyzer.meters_per_pixel
        
        all_results = {}
        
        for path_name, path_dict in paths_data.items():
            print(f"  → Simulating path: {path_name}")
            
            mining_polygon = path_dict['polygon']
            centerline = path_dict['centerline']
            buffer_zone = mining_polygon.buffer(buffer_distance_m)
            
            # Recalculate accurate counts using .intersects() to catch clipping
            direct_counts = {c: 0 for c in polygons_by_class.keys()}
            indirect_counts = {c: 0 for c in polygons_by_class.keys()}
            
            for class_name, poly_list in polygons_by_class.items():
                for poly in poly_list:
                    if poly.intersects(mining_polygon):
                        direct_counts[class_name] += 1
                    elif poly.intersects(buffer_zone):
                        indirect_counts[class_name] += 1
            
            def plot_loss(ax: plt.Axes, mode: str) -> None:
                # 1. Hazard Buffer
                try:
                    buf_coords = np.array(buffer_zone.exterior.coords)
                    ax.add_patch(mpatches.Polygon(buf_coords, fill=True, facecolor='#FFA500', 
                                                  edgecolor='none', alpha=0.15, zorder=1))
                except Exception: pass
                
                # 2. Mining Footprint & Centerline
                try:
                    mine_coords = np.array(mining_polygon.exterior.coords)
                    ax.add_patch(mpatches.Polygon(mine_coords, fill=True, facecolor='#FF4B4B', 
                                                  edgecolor='none', alpha=0.25, zorder=2))
                    ax.plot(mine_coords[:, 0], mine_coords[:, 1], color='#FF4B4B', 
                            linewidth=2.0, linestyle='--', zorder=3)
                            
                    line_coords = np.array(centerline.coords)
                    ax.plot(line_coords[:, 0], line_coords[:, 1], color='#FF4B4B', 
                            linewidth=1.0, linestyle='-.', alpha=0.8, zorder=3)
                except Exception: pass
                
                # 3. Draw classified polygons (Green = Resource, Red = Hazard)
                for class_name, poly_list in polygons_by_class.items():
                    is_resource = 'nodule' in class_name.lower() or 'rock' in class_name.lower()
                    
                    for poly in poly_list:
                        try:
                            # Static coloring based on object type
                            if is_resource:
                                color = '#3DDC84' # Green (Target)
                            else:
                                color = '#FF4B4B' # Red (Hazard: Organism/Obstruction)
                                
                            coords_m = np.array(poly.exterior.coords) * meters_per_px
                            patch = mpatches.Polygon(coords_m, fill=True, facecolor=(*matplotlib.colors.to_rgb(color), 0.4), 
                                                     edgecolor=color, linewidth=1.2, zorder=4)
                            ax.add_patch(patch)
                        except Exception: continue
                
                ax.set_title(f"Biological Loss Simulation: {path_name.replace('_', ' ').title()}\nTrack: {corridor_width_m}m | Buffer: {buffer_distance_m}m", fontsize=11, fontweight='bold')

            map_paths = self._render_and_save(f'biological_loss_{path_name}', plot_loss)
            
            chart_path = str(self.output_dir / f"biological_loss_{path_name}_chart.png")
            # Pass our custom, accurately calculated dictionaries directly to the chart plotter
            self._plot_biological_loss_chart(direct_counts, indirect_counts, chart_path, f"Impacts: {path_name.replace('_', ' ').title()}")
            map_paths['chart'] = chart_path
            
            all_results[path_name] = map_paths
            
        return all_results




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
    
    # try:
    #     results['nearest_neighbor_distance'] = viz.visualize_nearest_neighbor_distance()
    # except Exception as e:
    #     print(f"[ERROR] NND visualization failed: {e}")
    #     results['nearest_neighbor_distance'] = None
    
    # try:
    #     results['passability_index'] = viz.visualize_passability_index()
    # except Exception as e:
    #     print(f"[ERROR] Passability visualization failed: {e}")
    #     results['passability_index'] = None

    # try:
    #     results['spatial_homogeneity'] = viz.visualize_spatial_homogeneity()
    # except Exception as e:
    #     print(f"[ERROR] Spatial homogeneity visualization failed: {e}")
    #     results['spatial_homogeneity'] = None
    
    # try:
    #     results['solidity_rugosity'] = viz.visualize_solidity_rugosity()
    # except Exception as e:
    #     print(f"[ERROR] Solidity visualization failed: {e}")
    #     results['solidity_rugosity'] = None
    
    # try:
    #     results['obb_directionality'] = viz.visualize_obb_directionality()
    # except Exception as e:
    #     print(f"[ERROR] OBB visualization failed: {e}")
    #     results['obb_directionality'] = None

    # # Bivariate Ripley's K (Invisible Halo)
    # try:
    #     results['bivariate_ripleys_k'] = viz.visualize_bivariate_ripleys_k()
    # except Exception as e:
    #     print(f"[ERROR] Bivariate Ripley's K visualization failed: {e}")
    #     results['bivariate_ripleys_k'] = None
    
    # Phase 3: Verticality Metrics (3D - require elevation)
    # try:
    #     results['protrusion'] = viz.visualize_protrusion()
    # except Exception as e:
    #     print(f"[ERROR] Protrusion visualization failed: {e}")
    #     results['protrusion'] = None
    
    # try:
    #     results['embedment_angle'] = viz.visualize_embedment_angle()
    # except Exception as e:
    #     print(f"[ERROR] Embedment Angle visualization failed: {e}")
    #     results['embedment_angle'] = None

    try:
        results['projected_biological_loss'] = viz.visualize_projected_biological_loss()
    except Exception as e:
        print(f"[ERROR] Biological loss visualization failed: {e}")
        results['projected_biological_loss'] = None
    
    print("\n" + "="*60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*60)
    
    # Save collected legend images and include mapping in results
    try:
        legends_map = viz._save_legends()
        results['legends'] = legends_map
    except Exception as e:
        print(f"[WARNING] Saving legends failed: {e}")
        results['legends'] = {}

    # Save recorded insets (histograms, roses) as separate images
    try:
        insets_map = viz._save_insets()
        results['insets'] = insets_map
    except Exception as e:
        print(f"[WARNING] Saving insets failed: {e}")
        results['insets'] = {}

    return results