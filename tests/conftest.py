"""A tiny synthetic CellDIVE slide (the *synthetic-fixture rule*: the suite must run without real data).

``make_synthetic_slide`` writes a slide folder in the real on-disk convention: two pyramidal OME-TIFF
markers (DAPI, PanCK), three HALO GeoJSON region layers (R1 and R2 overlap, R3 stands alone), QC CSVs
for R1 and R2 (every 4th nucleus flagged Wrong) and one tiny fragment contour in R1 with no QC row.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import tifffile

SLIDE = "SLIDE-9999"
PIXEL_SIZE = 0.5
H, W = 600, 800
REGION_BOXES = {1: (50, 40, 350, 240), 2: (300, 120, 600, 320), 3: (450, 350, 750, 550)}   # x0,y0,x1,y1 (nucleus placement)
UNANNOTATED_BOX = (60, 350, 300, 560)   # tissue with nuclei that nobody annotated (-> unlabeled patches)
QC_REGIONS = (1, 2)


def _ring(cx: float, cy: float, rx: float, ry: float, n: int = 28) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)
    return np.round(pts).astype(np.int64)


def _pyramid(img: np.ndarray, levels: int = 3):
    out = [img]
    for _ in range(levels - 1):
        a = out[-1]
        h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
        out.append(a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(img.dtype))
    return out


def write_ome_pyramid(path: Path, img: np.ndarray, pixel_size: float = PIXEL_SIZE, tile: int = 64, levels: int = 3):
    lv = _pyramid(img, levels)
    with tifffile.TiffWriter(path, ome=True) as tw:
        tw.write(lv[0], subifds=len(lv) - 1, tile=(tile, tile), compression="lzw",
                 metadata={"axes": "YX", "PhysicalSizeX": pixel_size, "PhysicalSizeXUnit": "µm",
                           "PhysicalSizeY": pixel_size, "PhysicalSizeYUnit": "µm"})
        for a in lv[1:]:
            tw.write(a, subfiletype=1, tile=(tile, tile), compression="lzw")


def make_synthetic_slide(root: Path, slide: str = SLIDE, seed: int = 0) -> dict:
    """Write the slide folder under ``root`` and return ground truth for the tests."""
    rng = np.random.default_rng(seed)
    sdir = root / slide
    (sdir / f"HALO_layers_{slide[-4:]}").mkdir(parents=True)
    (sdir / f"QC_for_slide-{slide[-4:]}").mkdir()
    (sdir / f"SS For  {slide}" / "images").mkdir(parents=True)

    dapi = (rng.normal(120, 15, (H, W)).clip(60, 300)).astype(np.float32)
    panck = (rng.normal(80, 10, (H, W)).clip(40, 200)).astype(np.float32)
    truth = {"regions": {}, "slide": slide, "pixel_size": PIXEL_SIZE, "shape": (H, W)}
    for rid, (x0, y0, x1, y1) in REGION_BOXES.items():
        rings = []
        for cy in range(y0 + 12, y1 - 12, 22):
            for cx in range(x0 + 12, x1 - 12, 22):
                jx, jy = rng.integers(-3, 4), rng.integers(-3, 4)
                rx, ry = rng.uniform(5, 8), rng.uniform(5, 8)
                rings.append(_ring(cx + jx, cy + jy, rx, ry))
        tiny_idx = None
        if rid == 1:                                   # a degenerate fragment with no QC row
            tiny_idx = len(rings)
            rings.append(np.array([[x0 + 5, y0 + 5], [x0 + 7, y0 + 5], [x0 + 6, y0 + 7]]))
        for r in rings:
            cv2.fillPoly(dapi, [r.reshape(-1, 1, 2).astype(np.int32)], float(rng.uniform(1500, 3000)))
            cv2.fillPoly(panck, [r.reshape(-1, 1, 2).astype(np.int32)], float(rng.uniform(300, 900)))
        feats = [{"type": "Feature",
                  "properties": {"object_type": "annotation", "classification": {"name": f"Region {rid}", "colorRGB": -16711936},
                                 "isLocked": False, "layer_name": f"Region {rid}"},
                  "geometry": {"type": "LineString", "coordinates": np.vstack([r, r[:1]]).astype(float).tolist()}}
                 for r in rings]
        gj = {"type": "FeatureCollection", "name": f"CellDIVE_{slide}_Region_{rid}",
              "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, "features": feats}
        (sdir / f"HALO_layers_{slide[-4:]}" / f"CellDIVE_{slide}_Region_{rid}.geojson").write_text(json.dumps(gj))

        wrong = []
        if rid in QC_REGIONS:
            rows = []
            for i, r in enumerate(rings):
                if i == tiny_idx:
                    continue
                flag = "Wrong" if i % 4 == 1 else np.nan
                if flag == "Wrong":
                    wrong.append(i + 1)
                area_px = float(cv2.contourArea(r.astype(np.float32)))
                rows.append({"Image Location": r"\\server\share\CellDIVE_SLIDE-9999_R0.afi", "Analysis Region": "field of view",
                             "Algorithm Name": "Nuc Seg Test", "Object Id": len(rows),
                             "XMin": int(r[:, 0].min() + rng.integers(0, 2)), "XMax": int(r[:, 0].max() + rng.integers(0, 2)),
                             "YMin": int(r[:, 1].min() + rng.integers(0, 2)), "YMax": int(r[:, 1].max() + rng.integers(0, 2)),
                             "Classifier Label": "Nuclei", "Region Perimeter (μm)": round(cv2.arcLength(r.astype(np.float32), True) * PIXEL_SIZE, 2),
                             "Region Area (μm²)": round(area_px * PIXEL_SIZE ** 2, 2), "Flag": flag, "Total Cells Flagged": np.nan})
            rows[0]["Total Cells Flagged"] = str(len(wrong))
            pd.DataFrame(rows).to_csv(sdir / f"QC_for_slide-{slide[-4:]}" / f"NT_{slide}_R0_Region_{rid} - x.csv.csv", index=False)
        allp = np.vstack(rings)
        truth["regions"][rid] = {"n": len(rings), "wrong_labels": wrong, "tiny_index": tiny_idx,
                                 "bbox": (int(allp[:, 0].min()), int(allp[:, 1].min()), int(allp[:, 0].max()) + 1, int(allp[:, 1].max()) + 1),
                                 "has_qc": rid in QC_REGIONS}

    # unannotated tissue: nuclei drawn into the images only, no GeoJSON, no QC
    ux0, uy0, ux1, uy1 = UNANNOTATED_BOX
    for cy in range(uy0 + 12, uy1 - 12, 22):
        for cx in range(ux0 + 12, ux1 - 12, 22):
            r = _ring(cx + rng.integers(-3, 4), cy + rng.integers(-3, 4), rng.uniform(5, 8), rng.uniform(5, 8))
            cv2.fillPoly(dapi, [r.reshape(-1, 1, 2).astype(np.int32)], float(rng.uniform(1500, 3000)))
            cv2.fillPoly(panck, [r.reshape(-1, 1, 2).astype(np.int32)], float(rng.uniform(300, 900)))
    truth["unannotated_box"] = UNANNOTATED_BOX
    dapi = cv2.GaussianBlur(dapi, (0, 0), 1.2).astype(np.uint16)
    panck = cv2.GaussianBlur(panck, (0, 0), 1.2).astype(np.uint16)
    write_ome_pyramid(sdir / f"{slide}_1.0.4_R000_DAPI__FINAL_F.ome.tif", dapi)
    write_ome_pyramid(sdir / f"{slide}_9.0.4_R000_FITC_PanCK-C11-AF488_FINAL_AFR_F.ome.tif", panck)
    (sdir / "not_a_channel.tif").write_bytes(b"")                       # must be reported, not used
    (sdir / f"SS For  {slide}" / "images" / "image1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    np.save(root / f"{slide}_DAPI.npy", dapi)
    np.save(root / f"{slide}_PanCK.npy", panck)
    truth["arrays"] = {"DAPI": root / f"{slide}_DAPI.npy", "PanCK": root / f"{slide}_PanCK.npy"}
    return truth


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("mxdata_root")
    truth = make_synthetic_slide(root)
    truth["root"] = root
    return truth


@pytest.fixture()
def synth_copy(synth, tmp_path):
    """A private, writable copy of the synthetic slide folder (for tests that break things)."""
    root = tmp_path / "root"
    shutil.copytree(synth["root"] / SLIDE, root / SLIDE)
    return root


SLIDE2 = "SLIDE-9998"


@pytest.fixture(scope="session")
def synth2(synth):
    """A second synthetic slide in the same root (same region layout, different pixels) for multi-slide tests."""
    truth = make_synthetic_slide(synth["root"], slide=SLIDE2, seed=1)
    truth["root"] = synth["root"]
    return truth


@pytest.fixture()
def cfg():
    from mxdata import DataConfig, LoaderConfig, NormalizationConfig, PatchConfig, SplitConfig
    return DataConfig(slides=(SLIDE,), markers=("DAPI", "PanCK"), min_area_px=20.0,
                      patches=PatchConfig(size=64, stride=48, min_fg=0.02, min_tissue=0.2, tissue_max_dim=256, rows_per_unit=4),
                      normalization=NormalizationConfig(sample_pixels=20_000, seed=1),
                      split=SplitConfig(val_regions=(3,)), loader=LoaderConfig(batch_size=4, num_workers=0))


@pytest.fixture()
def dataset(synth, cfg, tmp_path):
    """A freshly sampled segmentation dataset of the synthetic slide."""
    from mxdata import sample_patches
    out = tmp_path / "dataset"
    return out, sample_patches(cfg, out, root=synth["root"])
