"""The downstream-analysis machinery (error taxonomy, marker tables, graphs) on small synthetic label maps."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mxseg.analysis import (
    boundary_f1,
    cell_marker_table,
    classify_errors,
    gate_phenotypes,
    knn_edges,
    neighborhood_composition,
    otsu_threshold,
)

ROOT = Path(__file__).resolve().parents[1]


def _maps():
    gt = np.zeros((60, 90), np.int32)
    gt[5:20, 5:20] = 1                                   # matched well
    gt[5:20, 30:45] = 2                                  # split into two predictions
    gt[30:45, 5:20] = 3                                  # merged with 4
    gt[30:45, 21:36] = 4
    gt[30:45, 60:75] = 5                                 # missed
    pred = np.zeros_like(gt)
    pred[5:20, 5:20] = 1                                 # == gt 1
    pred[5:20, 30:37] = 2                                # left half of gt 2
    pred[5:20, 38:45] = 3                                # right half of gt 2
    pred[30:45, 5:36] = 4                                # covers gt 3 + 4
    pred[50:58, 80:88] = 5                               # spurious
    return gt, pred


def test_error_taxonomy_deepcell_classes():
    gt, pred = _maps()
    df, gained = classify_errors(gt, pred)
    cls = dict(zip(df.gt_id, df.cls))
    assert cls == {1: "correct", 2: "split", 3: "merge", 4: "merge", 5: "missed"}
    assert df[df.gt_id == 1].iou.iloc[0] == 1.0
    assert gained == [5]                                 # the spurious prediction
    ign = np.zeros_like(gt, bool)
    ign[gt == 5] = True                                  # QC-wrong nucleus disappears from the analysis
    df2, gained2 = classify_errors(gt, pred, ign)
    assert 5 not in set(df2.gt_id)
    assert gained2 == [5]                                # spurious pred is NOT on the wrong nucleus -> still gained
    pred3 = pred.copy()
    pred3[50:58, 80:88] = 0
    pred3[30:45, 60:75] = 5                              # prediction sitting ON the QC-wrong nucleus
    _, gained3 = classify_errors(gt, pred3, ign)
    assert gained3 == []                                 # excluded per the benchmark ignore convention


def test_error_taxonomy_weak_and_catastrophe():
    gt = np.zeros((40, 60), np.int32)
    gt[5:20, 5:20] = 1                                   # weak one-to-one (pred shifted, IoU < 0.6)
    gt[25:35, 5:29] = 2                                  # catastrophe pair: 2 and 3 vs two preds crossing both
    gt[25:35, 31:55] = 3
    pred = np.zeros_like(gt)
    pred[12:27, 5:20] = 1                                # IoU with gt1 ~ 0.36 -> weak
    pred[25:30, 5:55] = 2                                # crosses gt 2 and 3
    pred[30:35, 5:55] = 3                                # crosses gt 2 and 3 too
    df, gained = classify_errors(gt, pred)
    cls = dict(zip(df.gt_id, df.cls))
    assert cls[1] == "missed"                            # DeepCell: weak 1-1 counts FN + FP
    assert cls[2] == "catastrophe" and cls[3] == "catastrophe"
    assert gained == []                                  # every prediction overlaps ground truth


def test_boundary_f1_perfect_and_shifted():
    gt, pred = _maps()
    assert boundary_f1(gt, pred, [(1, 1)]) == 1.0
    shifted = np.roll(pred, 1, axis=1)
    assert 0.5 < boundary_f1(gt, shifted, [(1, 1)], tol_px=2) <= 1.0


def test_marker_table_and_gating():
    gt, _ = _maps()
    markers = np.zeros((2, *gt.shape), np.float32)
    markers[0][gt == 1] = 100.0                          # PanCK on cell 1
    markers[1][gt == 3] = 80.0                           # CD45 on cell 3
    t = cell_marker_table(gt, markers, ["PanCK", "CD45"])
    assert list(t.id) == [1, 2, 3, 4, 5] and t.PanCK.iloc[0] == 100.0 and t.CD45.iloc[2] == 80.0
    assert abs(t.cy.iloc[0] - 12.0) < 0.6 and abs(t.cx.iloc[0] - 12.0) < 0.6
    ph = gate_phenotypes(t, {"PanCK": 50, "CD45": 40})
    assert list(ph) == ["Epithelial", "Other", "Immune", "Other", "Other"]
    thr = otsu_threshold(np.array([1.0] * 50 + [100.0] * 50))
    assert 1.0 < thr < 100.0


def test_graph_and_composition():
    cy = np.array([0, 0, 10, 10.0])
    cx = np.array([0, 5, 0, 5.0])
    edges = knn_edges(cy, cx, k=1)
    assert (0, 1) in edges and (2, 3) in edges
    comp = neighborhood_composition(["A", "B", "A", "B"], {(0, 1), (2, 3)}, ["A", "B"])
    assert comp.shape == (4, 2) and comp[0, 1] == 1.0 and comp[1, 0] == 1.0
    assert np.allclose(comp.sum(1), 1.0)


def test_classify_errors_frame_columns():
    gt, pred = _maps()
    df, _ = classify_errors(gt, pred)
    assert isinstance(df, pd.DataFrame) and set(df.columns) == {"gt_id", "cls", "pred_id", "iou", "cy", "cx", "area"}


def test_panel_phenotypes(tmp_path):
    """The panel config: marker roles / colors, and the biologist's one-marker-per-phenotype hierarchy gated top-down."""
    from mxseg.analysis import assign_phenotypes, load_panel, phenotype_gates
    y = tmp_path / "panel.yaml"
    y.write_text("""
nuclear: DAPI
envelope: LAP2
markers:
  DAPI:  {role: nuclei, color: "#0000FF"}
  LAP2:  {role: nuclear envelope, color: "#00FFFF"}
  PanCK: {role: epithelium, color: "#00FF00"}
  uPAR:  {role: uPAR, color: "#FF8800"}
  CD45:  {role: immune, color: "#FF0000"}
  CD3:   {role: T cell, color: "#FFA500"}
  CD4:   {role: helper T cell, color: "#FFFF00"}
phenotypes:
  - {name: Epithelial, marker: PanCK}
  - {name: uPAR+ epithelial, marker: uPAR, parent: Epithelial}
  - {name: T cell, marker: CD3}
  - {name: CD4+ T cell, marker: CD4, parent: T cell}
  - {name: Immune, marker: CD45}
gates: {PanCK: 5, uPAR: 5, CD45: 5, CD3: 5, CD4: 5}
""")
    panel = load_panel(y)
    assert panel.nuclear == "DAPI" and panel.envelope == "LAP2" and panel.colors["PanCK"] == "#00FF00"
    assert [p.name for p in panel.chain("CD4+ T cell")] == ["T cell", "CD4+ T cell"] and panel.lineage("uPAR+ epithelial") == "Epithelial"
    assert panel.lineages == ["Epithelial", "T cell", "Immune"] and panel.gate_markers == ["PanCK", "uPAR", "CD3", "CD4", "CD45"]
    cells = pd.DataFrame({"PanCK": [10, 10, 0, 0, 0, 0], "uPAR": [10, 0, 0, 0, 0, 10], "CD45": [0, 0, 10, 10, 0, 0],
                          "CD3": [0, 0, 10, 0, 0, 0], "CD4": [0, 0, 10, 10, 0, 0]})
    thr = phenotype_gates(cells, panel)                                             # fixed gates from the panel
    ph, lin = assign_phenotypes(cells, panel, thr)
    assert list(ph) == ["uPAR+ epithelial", "Epithelial", "CD4+ T cell", "Immune", "Other", "Other"]   # deepest positive chain;
    assert list(lin) == ["Epithelial", "Epithelial", "T cell", "Immune", "Other", "Other"]              # CD4 without CD3 is not a T cell
    shipped = load_panel(ROOT / "configs" / "panel.yaml")
    assert len(shipped.markers) == 9 and shipped.lineage("Treg") == "T cell"
    assert isinstance(shipped.gates, dict) and set(shipped.gates) >= set(shipped.gate_markers)      # fixed gates in slide-normalized units, never a histogram heuristic
    assert all(0 < float(v) < 1 for v in shipped.gates.values())
    norms = {m: (10.0, 1010.0) for m in shipped.gate_markers}
    raw = phenotype_gates(pd.DataFrame(), shipped, norms=norms)                                   # normalized -> this slide's counts
    assert all(abs(raw[m] - (10.0 + float(shipped.gates[m]) * 1000.0)) < 1e-6 for m in shipped.gate_markers)
    bad = tmp_path / "bad.yaml"
    bad.write_text("markers: {DAPI: {color: '#000'}}\nphenotypes: [{name: X, marker: DAPI, parent: Nope}]\n")
    with pytest.raises(ValueError, match="unknown parent"):
        load_panel(bad)
