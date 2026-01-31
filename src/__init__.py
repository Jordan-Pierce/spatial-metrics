"""
Spatial Metrics - A comprehensive spatial analysis toolkit for deep-sea mining.

This package provides tools for computing spatial metrics from geospatial data,
including orthorectified imagery, elevation rasters, masks, and GeoJSON annotations.

Main Classes:
    SpatialMetricsAnalyzer: Core class for computing all spatial metrics

Convenience Functions:
    analyze: Quick analysis with automatic report generation

Example:
    >>> from spatial_metrics import SpatialMetricsAnalyzer, analyze
    >>> 
    >>> # Using the class
    >>> analyzer = SpatialMetricsAnalyzer(
    ...     mask_path="nodules.tif",
    ...     geojson_path="annotations.geojson"
    ... )
    >>> summary_df, report = analyzer.generate_report()
    >>> 
    >>> # Using the convenience function
    >>> summary_df, report = analyze(
    ...     mask_path="nodules.tif",
    ...     geojson_path="annotations.geojson",
    ...     output_path="results.json"
    ... )
"""

from .analyzer import SpatialMetricsAnalyzer, analyze

__version__ = "0.1.0"
__author__ = "Jordan Pierce"
__all__ = ["SpatialMetricsAnalyzer", "analyze"]
