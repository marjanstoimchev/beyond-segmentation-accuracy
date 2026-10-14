"""mxdata — data processing and loading for CellDIVE multiplex slides with HALO nucleus annotations.

    io            files of a slide, CellDIVE file names, Box (level-0 px), MxSlide (lazy pyramid reads)
    annotations   HALO GeoJSON contours, QC table join, rasterization, QC flags as data
    pipeline      ingest -> validate -> plan -> stats -> sample  (the fixed segmentation dataset)
    splits        overlap-aware / leave-slide-out splits
    data          PatchSource | LiveSource -> RegionDataset -> DataLoader   (needs torch; imported lazily)
    cli           mxdata report | validate | sample | dataset | config
    phenotyping   the biologist's phenotype labels of a slide, unified in <slide>/phenotyping/ (areas, labels, tagged nuclei)

Quick start::

    from mxdata import load_config, build_datasets, build_loaders
    bundle = build_datasets(load_config("configs/slide-0235.yaml"))     # reads $MXDATA_ROOT/dataset
    train_loader, val_loader = build_loaders(bundle)
"""
from . import phenotyping
from ._version import __version__
from .annotations import (
    NucleusFlag,
    RegionAnnotation,
    apply_qc_policy,
    load_halo_geojson,
    load_qc_csv,
    load_regions,
    match_qc_to_polygons,
    nuclei_table,
    overlap_table,
    rasterize_polygons,
    regions_table,
)
from .config import (
    AugmentConfig,
    DataConfig,
    LoaderConfig,
    NormalizationConfig,
    PatchConfig,
    SplitConfig,
    config_from_dict,
    load_config,
    save_resolved,
    to_dict,
)
from .errors import AnnotationError, ConfigError, DatasetError, MxDataError, SlideError, ValidationError
from .io import (
    DATA_ROOT,
    DEFAULT_DATASET,
    Box,
    ChannelInfo,
    MxSlide,
    SlidePaths,
    parse_channel_filename,
)
from .log import setup_logging
from .pipeline import (
    IngestReport,
    NormStats,
    SampleResult,
    TissueGate,
    ValidationReport,
    compute_norm_stats,
    dataset_summary,
    ingest,
    plan_units,
    sample_patches,
    tile_grid,
    validate,
)
from .splits import RegionInfo, assign_splits, overlap_groups, split_regions

_DATA = {"RegionDataset", "LiveSource", "PatchSource", "RegionSample", "DataBundle", "build_datasets", "build_loaders",
         "make_dataloader", "collate_samples", "Normalizer", "Augment"}


def __getattr__(name):
    """The torch-dependent data layer is imported on first use, so the core stays importable without torch."""
    if name in _DATA:
        from . import data
        return getattr(data, name)
    raise AttributeError(f"module 'mxdata' has no attribute {name!r}")


__all__ = [
    "phenotyping","__version__", "setup_logging",
           "DATA_ROOT", "DEFAULT_DATASET", "SlidePaths", "ChannelInfo", "parse_channel_filename", "Box", "MxSlide",
           "NucleusFlag", "RegionAnnotation", "load_regions", "load_halo_geojson", "load_qc_csv", "match_qc_to_polygons",
           "rasterize_polygons", "apply_qc_policy", "regions_table", "overlap_table", "nuclei_table",
           "DataConfig", "PatchConfig", "NormalizationConfig", "SplitConfig", "AugmentConfig", "LoaderConfig",
           "load_config", "config_from_dict", "to_dict", "save_resolved",
           "MxDataError", "ConfigError", "SlideError", "AnnotationError", "ValidationError", "DatasetError",
           "IngestReport", "ValidationReport", "NormStats", "SampleResult", "TissueGate", "ingest", "validate", "plan_units",
           "tile_grid", "compute_norm_stats", "sample_patches", "dataset_summary",
           "RegionInfo", "overlap_groups", "split_regions", "assign_splits", *sorted(_DATA)]
