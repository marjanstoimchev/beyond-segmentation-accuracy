"""The stages, each handing a checked contract to the next:

    ingest    discover + open + read metadata          -> (paths, MxSlide, IngestReport)
    validate  refuse / flag what would corrupt results -> ValidationReport (annotates, never deletes)
    plan      decide the work before doing any of it   -> region units, patch grids (pure; dry-run countable)
    stats     per-marker normalization from tissue     -> NormStats (seeded, JSON)
    sample    the fixed segmentation dataset           -> tissue-gated image/mask patches + provenance tables
"""
from .ingest import IngestReport, ingest
from .patches import (
           INSTANCE_COLUMNS,
           KINDS,
           PATCH_COLUMNS,
           PatchSpec,
           UnlabeledUnit,
           plan_labeled,
           plan_unlabeled,
           read_image,
           read_patch,
)
from .plan import RegionUnit, Tile, plan_units, tile_grid
from .prepare import ensure_stats, prepare, write_json
from .sample import (
           DATASET_FORMAT,
           SampleResult,
           dataset_summary,
           read_dataset_json,
           rebuild_tables,
           sample_patches,
)
from .stats import NormStats, compute_norm_stats
from .tissue import TissueGate, otsu_threshold, tissue_mask
from .validate import Issue, ValidationReport, validate

__all__ = ["ingest", "IngestReport", "validate", "ValidationReport", "Issue", "prepare", "ensure_stats", "write_json",
           "plan_units", "RegionUnit", "Tile", "tile_grid", "compute_norm_stats", "NormStats",
           "TissueGate", "otsu_threshold", "tissue_mask",
           "sample_patches", "SampleResult", "dataset_summary", "rebuild_tables", "read_dataset_json", "DATASET_FORMAT", "PatchSpec", "UnlabeledUnit", "plan_labeled",
           "plan_unlabeled", "read_image", "read_patch", "PATCH_COLUMNS", "INSTANCE_COLUMNS", "KINDS"]
