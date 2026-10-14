"""Training histories: what ``mxseg train`` saves under ``runs/<name>/`` (Lightning's CSV log, resolved config, summary).

Lightning writes one sparse CSV row per logging event; ``load_history`` splits it into a per-step table (training loss) and
a per-epoch table (epoch losses, validation loss, instance metrics, learning rate). Readable while a run is still going.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml


@dataclass(frozen=True)
class RunHistory:
    name: str
    path: Path
    config: dict            # resolved_config.yaml
    summary: dict           # summary.json ({} while the run is still training)
    steps: pd.DataFrame     # per logged train step: step, epoch, train/loss, train/flow, train/prob
    epochs: pd.DataFrame    # per epoch: train/*, val/*, lr — NaN where a metric was not computed that epoch

    @property
    def val_metrics(self) -> pd.DataFrame:
        """Epochs on which the instance metrics (val/f1 ...) were computed."""
        return self.epochs.dropna(subset=["val/f1"]) if "val/f1" in self.epochs else self.epochs.iloc[0:0]

    @property
    def monitor(self) -> str:
        """The validation quantity the run keeps its weights by (``val.monitor``; ``val/loss`` unless configured otherwise)."""
        return (self.config.get("val") or {}).get("monitor", "val/loss")

    @property
    def best_epoch(self) -> Optional[int]:
        """The epoch whose weights the run keeps: the minimum of the monitored loss (or the maximum of a monitored metric)."""
        mon = self.monitor
        if mon not in self.epochs or not self.epochs[mon].notna().any():
            return None
        s = self.epochs.dropna(subset=[mon])
        return int(s.loc[s[mon].idxmin() if mon.endswith("loss") else s[mon].idxmax(), "epoch"])


def _rename(c: str) -> str:
    return "lr" if c.startswith("lr-") else c.removesuffix("_step").removesuffix("_epoch")


def load_history(run_dir) -> RunHistory:
    path = Path(run_dir)
    csv = path / "logs" / "metrics.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} not found: not a run directory written by `mxseg train`")
    raw = pd.read_csv(csv)
    known = raw.dropna(subset=["epoch"])[["step", "epoch"]].drop_duplicates("step").sort_values("step")
    raw = pd.merge_asof(raw.sort_values("step", kind="stable"), known.rename(columns={"epoch": "_epoch"}), on="step", direction="forward")
    raw["epoch"] = raw["epoch"].fillna(raw.pop("_epoch"))        # rows logged without an epoch (learning rate) -> the epoch that starts there
    step_cols = [c for c in raw.columns if c.endswith("_step")]
    epoch_cols = [c for c in raw.columns if c not in step_cols and c not in ("epoch", "step")]
    steps = raw.dropna(subset=step_cols, how="all")[["step", "epoch", *step_cols]].rename(columns=_rename)
    epochs = (raw.dropna(subset=epoch_cols, how="all").dropna(subset=["epoch"]).drop(columns=["step", *step_cols])
              .groupby("epoch").max().rename(columns=_rename).reset_index())        # one row per epoch
    epochs["epoch"] = epochs["epoch"].astype(int)
    cfg_file, sum_file = path / "resolved_config.yaml", path / "summary.json"
    config = yaml.safe_load(cfg_file.read_text()) if cfg_file.is_file() else {}
    summary = json.loads(sum_file.read_text()) if sum_file.is_file() else {}
    return RunHistory(path.name, path, config, summary, steps, epochs)


def list_runs(runs_dir, names: Optional[List[str]] = None) -> List[RunHistory]:
    """Every run under ``runs_dir`` that has started logging (optionally restricted to ``names``)."""
    dirs = sorted(p for p in Path(runs_dir).iterdir() if (p / "logs" / "metrics.csv").is_file())
    return [load_history(p) for p in dirs if names is None or p.name in names]


def runs_table(hists: List[RunHistory]) -> pd.DataFrame:
    """One row per run: configuration that matters, progress, the kept epoch (``best_epoch``, the monitored optimum) with its
    validation loss and F1, the maximum F1 seen, and the final F1."""
    rows = []
    for h in hists:
        model = h.config.get("model", "cellpose")
        if not isinstance(model, str):                                     # not a fine-tuning run (KODIAK pretraining: model is a section)
            continue
        sec = h.config.get(model, {})
        variant = sec.get("train_blocks") if model == "cellpose" else sec.get("encoder")     # what identifies the run within its model family
        vm = h.val_metrics
        be = h.best_epoch
        at = h.epochs[h.epochs["epoch"] == be].iloc[0] if be is not None else None

        def val(col):                                                     # a quantity of the kept epoch, if it was computed then
            return float(at[col]) if at is not None and col in at.index and at[col] == at[col] else None
        rows.append({"run": h.name, "model": model, "variant": variant, "lr": (h.config.get("optim") or {}).get("lr", sec.get("lr")),
                     "batch": h.config.get("batch_size"),
                     "epochs": h.config.get("epochs"), "epochs_done": len(h.epochs), "finished": bool(h.summary),
                     "train_loss_last": h.epochs["train/loss"].dropna().iloc[-1] if "train/loss" in h.epochs and h.epochs["train/loss"].notna().any() else None,
                     "val_loss_last": h.epochs["val/loss"].dropna().iloc[-1] if "val/loss" in h.epochs and h.epochs["val/loss"].notna().any() else None,
                     "monitor": h.monitor, "best_epoch": be, "best_val_loss": val("val/loss"), "best_val_f1": val("val/f1"),
                     "max_val_f1": vm["val/f1"].max() if len(vm) else None, "final_val_f1": vm["val/f1"].iloc[-1] if len(vm) else None})
    return pd.DataFrame(rows).set_index("run")
