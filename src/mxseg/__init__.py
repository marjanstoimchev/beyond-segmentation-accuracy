"""mxseg — model baselines for nucleus instance segmentation on top of the ``mxdata`` dataset.

    baselines/   pluggable segmenters behind one interface (pretrained Cellpose-SAM, InstanSeg, trained nets later)
    metrics      instance-level scoring (matched F1 / AP / PQ on owned nuclei, QC-Wrong ignored) + pixel Dice
    evaluate     run a baseline over a split, store results under results/<split>/<name>/
    cli          mxseg eval | compare
"""
import os as _os

_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # SAM2's NMS has no MPS kernel; fall back to CPU for that op

from ._version import __version__
from .baselines import BASELINES, Baseline, get_baseline
from .evaluate import compare_results, read_prediction, run_baseline, stored_predictions
from .metrics import IOU_THRESHOLDS, aggregate, instance_outcomes, score_patch

__all__ = ["__version__", "BASELINES", "Baseline", "get_baseline", "run_baseline", "compare_results", "score_patch", "instance_outcomes", "read_prediction", "stored_predictions", "aggregate", "IOU_THRESHOLDS"]
