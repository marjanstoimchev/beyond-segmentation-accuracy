"""The biologist QC table exported from HALO.

One row per nucleus with an integer bbox (``XMin/XMax/YMin/YMax``) that equals the contour bbox up to
a +0/+1 px rounding, a ``Flag`` column (``"Wrong"`` or blank = correct) and a run summary in
``Total Cells Flagged``. Rows are matched to contours by nearest bbox (L-inf <= 1 px), one-to-one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from ..errors import AnnotationError

QC_REQUIRED_COLUMNS = ("Object Id", "XMin", "XMax", "YMin", "YMax", "Flag")
QC_WRONG = "Wrong"


def load_qc_csv(path: "str | Path") -> pd.DataFrame:
    """Read a HALO QC export and check its schema. Columns are kept verbatim."""
    p = Path(path)
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        raise AnnotationError(f"cannot read QC csv {p}: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in QC_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AnnotationError(f"{p.name}: QC table is missing column(s) {missing}",
                              hint=f"columns present: {list(df.columns)}")
    for c in ("XMin", "XMax", "YMin", "YMax"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise AnnotationError(f"{p.name}: column {c!r} is not numeric")
    return df


def qc_summary(df: pd.DataFrame) -> dict:
    flag = df["Flag"].fillna("").astype(str).str.strip()
    n_wrong = int((flag == QC_WRONG).sum())
    other = sorted(set(flag.unique()) - {"", QC_WRONG})
    return {"n_rows": int(len(df)), "n_wrong": n_wrong, "n_ok": int((flag == "").sum()),
            "pct_wrong": round(100.0 * n_wrong / max(len(df), 1), 2), "other_flags": other}


def match_qc_to_polygons(rings: Sequence[np.ndarray], qc: pd.DataFrame, tol: float = 1.0) -> Tuple[np.ndarray, dict]:
    """Assign each QC row to the contour with the nearest bbox (L-inf <= ``tol`` px), one-to-one.

    Returns ``(qc_row_of_polygon, stats)``: ``qc_row_of_polygon[i]`` is the QC row index for contour ``i``,
    or ``-1`` when unmatched. ``stats`` counts matched / unmatched on both sides.
    """
    from scipy.spatial import cKDTree

    n = len(rings)
    out = np.full(n, -1, dtype=np.int64)
    if n == 0 or len(qc) == 0:
        return out, {"matched": 0, "unmatched_qc": int(len(qc)), "unmatched_polys": n, "ambiguous": 0, "max_dist": None}
    pb = np.array([[r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max()] for r in rings], dtype=np.float64)
    qb = qc[["XMin", "XMax", "YMin", "YMax"]].to_numpy(dtype=np.float64)
    dist, idx = cKDTree(pb).query(qb, p=np.inf)
    ok = dist <= tol
    taken = np.zeros(n, dtype=bool)
    dup = 0
    for q in np.argsort(dist, kind="stable"):          # greedy one-to-one, closest first
        if not ok[q]:
            continue
        p = int(idx[q])
        if taken[p]:
            dup += 1
            continue
        taken[p] = True
        out[p] = q
    return out, {"matched": int(taken.sum()), "unmatched_qc": int(len(qc) - taken.sum()),
                 "unmatched_polys": int(n - taken.sum()), "ambiguous": dup,
                 "max_dist": float(dist[ok].max()) if ok.any() else None}
