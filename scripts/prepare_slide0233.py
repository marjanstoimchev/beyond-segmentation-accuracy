#!/usr/bin/env python
"""Convert the SLIDE-0233 delivery into the layout mxdata expects, so the slide can be sampled and scored.

Slides 0235/0236 arrive as HALO_layers_<id>/<...>Region_<k>.geojson plus QC_for_slide-<id>/*Region_<k>*.csv.
SLIDE-0233 arrived differently: ONE HALO LineString export covering all four immune-rich FOVs (duplicated under
several names), and four per-FOV review CSVs whose review column is called "ground truths"/"ground_truths" rather
than "Flag", with values in lower case. Nothing is deleted or moved: this writes two new folders beside the originals.

    conda activate dl
    python scripts/prepare_slide0233.py [--dry-run]
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
import numpy as np, pandas as pd
from mxdata import DATA_ROOT

SLIDE, SID = "SLIDE-0233", "0233"
EXPORT = "CellDIVE_SLIDE-0233_R2_IMMUNE_RICH.geojson"      # all six copies are byte-identical; any one will do


def contour_bboxes(feats):
    out = []
    for f in feats:
        pts = []
        def walk(o):
            if o and isinstance(o[0], (int, float)) and len(o) >= 2:
                pts.append(o); return
            for k in o: walk(k)
        walk(f["geometry"]["coordinates"])
        a = np.asarray(pts, float)
        out.append((a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()))
    return np.asarray(out)


def main(dry: bool):
    S = Path(DATA_ROOT) / SLIDE
    src = S / EXPORT
    if not src.is_file():
        src = S / "original_delivery" / "CellDIVE_SLIDE-0233_R1_IMMUNE_RICH.geojson"
    feats = json.loads(src.read_text())["features"]
    B = contour_bboxes(feats)
    cx, cy = (B[:, 0] + B[:, 1]) / 2, (B[:, 2] + B[:, 3]) / 2
    print(f"export {src.name}: {len(feats)} contours", flush=True)

    qc_dir, halo_dir = S / f"QC_for_slide-{SID}", S / f"HALO_layers_{SID}"
    if not dry:
        qc_dir.mkdir(exist_ok=True); halo_dir.mkdir(exist_ok=True)

    used = np.zeros(len(feats), bool)
    for p in sorted((S / "original_delivery").glob("*.csv")):
        k = int(re.search(r"Region_(\d+)", p.name).group(1))
        d = pd.read_csv(p)
        col = [c for c in d.columns if "ground" in c.lower()][0]
        # the loader needs a Flag column with "Wrong" or blank; this delivery uses lower case and stray whitespace
        flag = d[col].fillna("").astype(str).str.strip().str.lower()
        d = d.drop(columns=[col])
        d["Flag"] = np.where(flag.eq("wrong"), "Wrong", "")
        d["Total Cells Flagged"] = int((d.Flag == "Wrong").sum())

        x0, x1, y0, y1 = d.XMin.min(), d.XMax.max(), d.YMin.min(), d.YMax.max()
        m = (cx >= x0 - 2) & (cx <= x1 + 2) & (cy >= y0 - 2) & (cy <= y1 + 2) & ~used
        used |= m
        sub = [feats[i] for i in np.nonzero(m)[0]]
        for f in sub:                                              # label the layer, as the 0236 exports do
            f.setdefault("properties", {})["classification"] = {"name": f"Region {k}", "colorRGB": -16711936}
            f["properties"]["layer_name"] = f"Region {k}"
        print(f"  R{k}: {len(d):4d} QC rows ({int((d.Flag=='Wrong').sum()):4d} wrong) -> {len(sub):4d} contours", flush=True)
        if not dry:
            (halo_dir / f"R{k}_HALO_{SLIDE}_Region_{k}.geojson").write_text(
                json.dumps({"type": "FeatureCollection", "features": sub}))
            d.to_csv(qc_dir / f"{SLIDE}_Region_{k}.csv", index=False)
    print(f"  {int((~used).sum())} contours fall in no FOV and are dropped", flush=True)
    print(("dry run, nothing written" if dry else f"wrote {halo_dir.name}/ and {qc_dir.name}/"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
