#!/usr/bin/env python
"""t-SNE of the token space, matched across feature sets (appendix A.6).

Illustrative only: the claim is carried by the leave-one-region-out probes, which this figure shows rather than
proves. Everything that could favour one side is held fixed -- the same tokens, the same labels, the same PCA width,
the same perplexity and seed -- so the panels differ only in the features underneath.

Three feature sets x two labellings: MxIF-SSL tokens, the DINOv3 initialization they continue from, and the marker
baseline (per-token mean and s.d. of the three input channels), coloured by reference niche and by cell lineage.

    conda activate dl
    python scripts/figure_tsne.py        # -> LaTex/figures/tsne/
"""
from __future__ import annotations
import os, shutil, tempfile, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mxdata import DEFAULT_DATASET, apply_qc_policy, ingest, load_config, load_regions
from mxdata.data import LiveSource
from mxdata.data.composite import render_channels
from mxseg.evaluate import split_regions_of
from mxseg.inputs import dataset_norms
from kodiak import Kodiak
from kodiak.components import DinoV3Encoder

REPO = Path(__file__).resolve().parents[1]
ANALYSIS, OUT = REPO / "outputs" / "analysis", REPO / "LaTex" / "figures" / "tsne"
RUN_NAME, SLIDE, GREEN, CKPT = "kodiak_mxif_vitb", "SLIDE-0236", "PanCK", "last.ckpt"
PCA_DIM, PERPLEXITY, SEED = 50, 30, 0
CLASSES, W_CN, NCN = ["Epithelial", "Immune", "Other"], 10, 4
NICHE_COLOR = {"Immune-infiltrated": "#C43E4E", "Epithelial core": "#2A78D6",
               "Interface": "#9A86E0", "Stromal / other": "#B9BDC4"}
LIN_COLOR = {"Epithelial": "#3FA96B", "Immune": "#E08A3C", "Other": "#C9CDD3"}
# the leave-one-region-out probe scores this figure illustrates (balanced accuracy, Section A.6.1)
PROBE = {("MxIF-SSL", "niche"): 0.529, ("DINOv3", "niche"): 0.483, ("markers", "niche"): 0.507,
         ("MxIF-SSL", "lineage"): 0.650, ("DINOv3", "lineage"): 0.625, ("markers", "lineage"): 0.698}
INK, MUTED = "#1A1C1F", "#6B7075"

matplotlib.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "pdf.fonttype": 42, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def windows(cy, cx, pheno):
    idx = {c: i for i, c in enumerate(CLASSES)}
    pts = np.column_stack([cy, cx])
    _, nn = cKDTree(pts).query(pts, k=min(W_CN, len(pts)))
    ph = np.array([idx[p] for p in pheno])
    comp = np.zeros((len(pts), len(CLASSES)))
    for i, row in enumerate(nn):
        for j in np.atleast_1d(row):
            comp[i, ph[j]] += 1
    return comp / np.maximum(comp.sum(1, keepdims=True), 1)


def encode_windows(x, fn, G, size=256, stride=128, batch=8):
    H, W = x.shape[1:]
    ys = sorted({*range(0, H - size + 1, stride), H - size}); xs = sorted({*range(0, W - size + 1, stride), W - size})
    wins = [(y, xx) for y in ys for xx in xs]; g = size // G
    hann = np.hanning(g + 2)[1:-1]; wgt = (np.outer(hann, hann) + 1e-3)[..., None].astype(np.float32)
    acc, cnt = None, np.zeros((H // G, W // G, 1), np.float32)
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]
        o = fn(torch.from_numpy(np.stack([x[:, y:y + size, xx:xx + size] for y, xx in chunk]))).numpy()
        if acc is None: acc = np.zeros((H // G, W // G, o.shape[-1]), np.float32)
        for j, (y, xx) in enumerate(chunk):
            ty, tx = y // G, xx // G
            acc[ty:ty + g, tx:tx + g] += o[j].reshape(g, g, -1) * wgt; cnt[ty:ty + g, tx:tx + g] += wgt
    return acc / cnt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "configs" / "dataset.yaml")
    torch.set_grad_enabled(False); torch.set_num_threads(max(2, (os.cpu_count() or 8) // 2))
    RUN = next(p for p in (REPO / "runs_v2" / RUN_NAME, REPO / "runs" / RUN_NAME) if p.is_dir())
    run_cfg = yaml.safe_load((RUN / "resolved_config.yaml").read_text()); M = run_cfg["model"]
    ARCH = dict(patch_size=M.get("vit_patch_size", 16), embed_dim=M.get("vit_embed_dim", 384), depth=M.get("vit_depth", 12),
                num_heads=M.get("vit_heads", 6), mlp_ratio=M.get("vit_mlp_ratio", 4.0),
                num_storage_tokens=M.get("num_storage_tokens", 4))
    DEC = dict(decoder_embed_dim=M.get("decoder_dim", 192), decoder_depth=M.get("decoder_depth", 4),
               decoder_num_heads=M.get("decoder_heads", 6))
    CHANNELS = (run_cfg["data"]["input"]["red"], GREEN, run_cfg["data"]["input"]["blue"])
    MARKERS = [m for m in CHANNELS if m]; G = ARCH["patch_size"]
    tmp = Path(tempfile.gettempdir()) / "tsne.ckpt"; shutil.copy2(RUN / "checkpoints" / CKPT, tmp)
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    kodiak = Kodiak(num_prototypes=M.get("num_prototypes", 128), projector_dim=M.get("projector_dim", 256),
                    img_size=M.get("image_size", 256), **ARCH, **DEC).eval()
    kodiak.load_state_dict({k.split("model.", 1)[1]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=False)
    dino = DinoV3Encoder(**ARCH).eval()
    ds = torch.load(REPO / M["pretrained_path"], map_location="cpu", weights_only=False)
    dino.load_state_dict(ds.get("model", ds), strict=True)
    ENC = {"MxIF-SSL": lambda xb: kodiak.teacher_encoder.forward_full(xb)[1],
           "DINOv3":   lambda xb: dino.forward_full(xb)[1]}
    print(f"encoders ready (epoch {int(ck['epoch'])})", flush=True)

    paths, mx, _ = ingest(SLIDE, None)
    NORMS = dataset_norms(DEFAULT_DATASET, cfg.slides, MARKERS)[SLIDE]
    IN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    IN_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    ids = sorted(split_regions_of(DEFAULT_DATASET, "test", cfg.slides).get(SLIDE, []))
    regions = {r.region_id: r for r in load_regions(paths, ids, cfg.min_area_px)}
    keys = [f"{SLIDE}_R{rid:02d}" for rid in regions]

    expert = pd.read_csv(ANALYSIS / "cells_expert.csv"); expert = expert[expert.region.isin(keys)]
    ecomp, frames = [], []
    for k in keys:
        e = expert[expert.region == k].reset_index(drop=True)
        ecomp.append(windows(e.cy.values, e.cx.values, list(e.pheno))); frames.append(e)
    km = MiniBatchKMeans(NCN, n_init=10, random_state=0).fit(np.vstack(ecomp))
    cc, NAME, left = km.cluster_centers_, {}, list(range(NCN))
    for nm, cls in (("Immune-infiltrated", "Immune"), ("Epithelial core", "Epithelial"), ("Interface", "Epithelial")):
        c = max(left, key=lambda i: cc[i, CLASSES.index(cls)]); NAME[c] = nm; left.remove(c)
    NAME[left[0]] = "Stromal / other"
    expert = pd.concat(frames, ignore_index=True); expert["niche"] = [NAME[c] for c in km.labels_]

    t0, FEAT, rows = time.perf_counter(), {t: [] for t in ENC}, []
    for rid, r in regions.items():
        key = f"{SLIDE}_R{rid:02d}"
        st = mx.read_region(r.box, markers=MARKERS, level=0)
        H, W = (st.shape[1] // G) * G, (st.shape[2] // G) * G; st = st[:, :H, :W]
        rgb = render_channels(st, MARKERS, NORMS, CHANNELS)
        x = ((rgb - IN_MEAN) / IN_STD).astype(np.float32)
        h, w = H // G, W // G
        m01 = np.stack([np.clip((st[c].astype(np.float32) - NORMS[m][0]) / max(NORMS[m][1] - NORMS[m][0], 1e-6), 0, 1)
                        for c, m in enumerate(MARKERS)])
        e = expert[expert.region == key]
        ty, tx = (e.cy // G).astype(int).values, (e.cx // G).astype(int).values
        ok = (ty < h) & (tx < w); e = e[ok].assign(t=ty[ok] * w + tx[ok])
        maj = lambda col: e.groupby("t")[col].agg(lambda s: s.value_counts().index[0])
        niche, lineage = np.full(h * w, "none", object), np.full(h * w, "none", object)
        niche[maj("niche").index] = maj("niche").values
        lineage[maj("pheno").index] = maj("pheno").values
        mean, sd = m01.reshape(3, h, G, w, G).mean((2, 4)), m01.reshape(3, h, G, w, G).std((2, 4))
        df = pd.DataFrame({"region": key, "n_cells": np.bincount(e.t, minlength=h * w), "niche": niche, "lineage": lineage})
        for c, m in enumerate(MARKERS):
            df[f"{m}_mean"], df[f"{m}_sd"] = mean[c].ravel(), sd[c].ravel()
        rows.append(df)
        for tag, fn in ENC.items():
            FEAT[tag].append(encode_windows(x, fn, G).reshape(h * w, -1))
    TOK = pd.concat(rows, ignore_index=True)
    X = {t: np.concatenate(v).astype(np.float32) for t, v in FEAT.items()}
    X["markers"] = TOK[[c for c in TOK.columns if c.endswith(("_mean", "_sd"))]].values.astype(np.float32)
    LAB = TOK.n_cells.values > 0
    print(f"{len(regions)} regions x {len(ENC)} encoders in {time.perf_counter()-t0:.0f}s | "
          f"{len(TOK)} tokens, {LAB.sum()} labelled", flush=True)

    unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    EMB = {}
    for tag in ("MxIF-SSL", "DINOv3", "markers"):
        Z = X[tag][LAB]
        Z = unit(Z) if tag != "markers" else (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-6)
        if Z.shape[1] > PCA_DIM:
            Z = PCA(PCA_DIM, random_state=SEED).fit_transform(Z)
        t1 = time.perf_counter()
        EMB[tag] = TSNE(2, perplexity=PERPLEXITY, init="pca", random_state=SEED, max_iter=1000).fit_transform(Z)
        print(f"  t-SNE {tag}: {Z.shape} in {time.perf_counter()-t1:.0f}s", flush=True)

    # does the 2-D picture actually rank the feature sets the way the held-out probe does? if not, the figure cannot
    # be read as evidence and should not be shown as if it were
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import cross_val_predict
    check = {}
    for tag in ("MxIF-SSL", "DINOv3", "markers"):
        for target in ("niche", "lineage"):
            y = TOK[target].values[LAB]
            pred = cross_val_predict(KNeighborsClassifier(30), EMB[tag], y, cv=5)
            check[f"{tag}|{target}"] = round(float(balanced_accuracy_score(y, pred)), 3)
    print("  2-D embedding kNN balanced accuracy (vs probe):", flush=True)
    for target in ("niche", "lineage"):
        row = {t: (check[f"{t}|{target}"], PROBE[(t, target)]) for t in ("MxIF-SSL", "DINOv3", "markers")}
        print(f"    {target:8s} " + "  ".join(f"{t}: 2D {a:.3f} / probe {b:.3f}" for t, (a, b) in row.items()), flush=True)
    np.savez(OUT / "tsne_embeddings.npz", **{f"emb_{t}": EMB[t] for t in EMB},
             niche=TOK["niche"].values[LAB].astype(str), lineage=TOK["lineage"].values[LAB].astype(str))

    PANEL = 1.62
    for target, cmap in (("niche", NICHE_COLOR), ("lineage", LIN_COLOR)):
        y = TOK[target].values[LAB]
        for tag in ("MxIF-SSL", "DINOv3", "markers"):
            v = EMB[tag]
            f = plt.figure(figsize=(PANEL, PANEL)); ax = f.add_axes([0, 0, 1, 1])
            for cls, col in cmap.items():
                m = y == cls
                if m.any():
                    ax.scatter(v[m, 0], v[m, 1], s=1.6, c=col, lw=0, alpha=0.75, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_edgecolor("#C6CBD2"); s.set_linewidth(0.5)
            ax.text(0.035, 0.965, f"probe {PROBE[(tag, target)]:.3f}", transform=ax.transAxes, va="top", ha="left",
                    fontsize=6.4, color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D9DDE2", lw=0.4))
            f.savefig(OUT / f"tsne_{target}_{tag.replace('/','').replace('-','')}.pdf", dpi=450); plt.close(f)
    for name, cmap in (("niche", NICHE_COLOR), ("lineage", LIN_COLOR)):
        lg = plt.figure(figsize=(5.2, 0.26))
        h = [plt.Line2D([], [], marker="o", ls="", ms=4.2, color=c) for c in cmap.values()]
        lg.legend(h, list(cmap), loc="center", ncol=len(cmap), fontsize=6.3, handlelength=1.0,
                  handletextpad=0.35, columnspacing=1.4, borderpad=0.0, borderaxespad=0.0, frameon=False)
        lg.savefig(OUT / f"tsne_legend_{name}.pdf", bbox_inches="tight", pad_inches=0.01); plt.close(lg)
    (OUT / "tsne.json").write_text(json.dumps(
        {"n_tokens": int(LAB.sum()), "perplexity": PERPLEXITY, "pca_dim": PCA_DIM, "seed": SEED, "knn_2d": check,
         "probe": {f"{k[0]}|{k[1]}": v for k, v in PROBE.items()}}, indent=1))
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
