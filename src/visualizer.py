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
        self.affine_transform: Optional[Any] = None
        self._cached_polygons: Optional[List[Polygon]] = None
        self._exemplar_indices: Optional[List[int]] = None
        
        self._load_orthomosaic()
    
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
        ax.set_ylim(y_max, y_min) # Y inverted for geospatial mapping
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
        extent = [0, max_x, max_y, 0] # BUG 1 FIXED: Mapped pixels to physical space
        
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
                max_radius = metric_results['max_passage_radius_m']
                
                # BUG 2 FIXED: Automatically extract physical coordinates for the max passage circle
                max_idx = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)
                max_y = max_idx[0] * self.analyzer.meters_per_pixel
                max_x = max_idx[1] * self.analyzer.meters_per_pixel
                
                circle = mpatches.Circle((max_x, max_y), max_radius, fill=False, edgecolor='#FF0000', linewidth=3)
                ax.add_patch(circle)
                ax.text(max_x, max_y - max_radius - 0.5, f'Max radius: {max_radius:.2f}m',
                        ha='center', fontsize=12, color='#FF0000', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
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
        # Completely separate rendering path for Grid layouts
        morphology_df = self.analyzer.calculate_morphology_stats()
        if len(morphology_df) == 0: 
            return {}
        
        indices = set()
        for percentile in np.linspace(0, 100, n_exemplars, endpoint=False):
            idx = int(len(morphology_df) * percentile / 100)
            sorted_df = morphology_df.sort_values('solidity')
            indices.add(sorted_df.index[idx])
        self._exemplar_indices = sorted(list(indices))
        polygons = self.analyzer._load_polygons()
        
        def draw_cell(ax, poly, solidity, mode):
            minx, miny, maxx, maxy = poly.bounds
            width = maxx - minx
            height = maxy - miny
            pad_x, pad_y = width * padding_fraction, height * padding_fraction
            
            # Apply padding to bounding box
            bbox_minx = minx - pad_x
            bbox_miny = miny - pad_y
            bbox_maxx = maxx + pad_x
            bbox_maxy = maxy + pad_y
            
            if mode == 'clean':
                ax.set_facecolor('white')
                poly_coords = np.array(poly.exterior.coords) * self.analyzer.meters_per_pixel
                ax.add_patch(mpatches.Polygon(poly_coords, fill=True, facecolor='gray', edgecolor='black', alpha=0.8))
                
            elif mode == 'overlay' and self.orthomosaic_array is not None:
                row_min, col_min = self._world_to_pixel(bbox_minx * self.analyzer.meters_per_pixel, 
                                                        bbox_miny * self.analyzer.meters_per_pixel)
                row_max, col_max = self._world_to_pixel(bbox_maxx * self.analyzer.meters_per_pixel, 
                                                        bbox_maxy * self.analyzer.meters_per_pixel)
                
                r_min, r_max = int(min(row_min, row_max)), int(max(row_min, row_max))
                c_min, c_max = int(min(col_min, col_max)), int(max(col_min, col_max))
                
                img_crop = self.orthomosaic_array[r_min:r_max, c_min:c_max]
                if img_crop.size > 0:
                    extent = [bbox_minx * self.analyzer.meters_per_pixel, bbox_maxx * self.analyzer.meters_per_pixel, 
                              bbox_maxy * self.analyzer.meters_per_pixel, bbox_miny * self.analyzer.meters_per_pixel]
                    ax.imshow(img_crop, origin='upper', extent=extent)
            
            hull = poly.convex_hull
            if hull.geom_type == 'Polygon':
                hull_coords = np.array(hull.exterior.coords) * self.analyzer.meters_per_pixel
                ax.add_patch(mpatches.Polygon(hull_coords, fill=False, edgecolor='#00FF00', linewidth=2.5))
            
            ax.set_xlim(bbox_minx * self.analyzer.meters_per_pixel, bbox_maxx * self.analyzer.meters_per_pixel)
            ax.set_ylim(bbox_maxy * self.analyzer.meters_per_pixel, bbox_miny * self.analyzer.meters_per_pixel)
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
    
    print("\n" + "="*60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*60)
    
    return results
