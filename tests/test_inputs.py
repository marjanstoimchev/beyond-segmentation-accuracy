"""The network input (mxseg.inputs): the channel assignment (a marker or blank per red / green / blue channel), the weights
round trip, the config block, and the whole train -> weights -> eval path on the synthetic slide (DAPI + PanCK)."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_render_channels_and_meta_roundtrip():
    from mxdata.errors import ConfigError
    from mxseg.inputs import InputConfig, InputSpec, input_meta
    stack = np.zeros((2, 8, 8), np.uint16)
    stack[0, :4], stack[1, :, :4] = 1000, 500                                        # PanCK top half, DAPI left half
    spec = InputSpec(("PanCK", None, "DAPI"), {"S": {"PanCK": (0.0, 1000.0), "DAPI": (0.0, 500.0)}})
    assert spec.markers == ["PanCK", "DAPI"] and spec.n_channels == 3
    x = spec.render(stack)                                                           # one slide: no name needed
    assert x.shape == (3, 8, 8) and x.dtype == np.float32
    assert np.allclose(x[:, 0, 0], [1, 0, 1]) and np.allclose(x[:, 7, 7], [0, 0, 0]) and x[1].max() == 0   # green stays blank
    with pytest.raises(ValueError, match="expects"):
        spec.render(stack[:1])
    nine = np.concatenate([stack[1:], np.full((1, 8, 8), 7, np.uint16), stack[:1]])  # a stored stack: DAPI, junk, PanCK
    assert np.allclose(spec.render_from(nine, ["DAPI", "CD3", "PanCK"], "S"), x)     # markers selected by name, case-insensitive
    two = InputSpec(spec.channels, {"S": spec.norms["S"], "T": spec.norms["S"]})
    with pytest.raises(ValueError, match="statistics"):
        two.render(stack)                                                            # two slides: the slide must be named
    assert two.render(stack, "T").shape == (3, 8, 8)
    assert InputSpec.from_meta(input_meta(spec)) == spec and "R PanCK · G — · B DAPI" in spec.describe()
    assert InputSpec.from_meta(input_meta(None)) is None and input_meta(None) == {"mode": "raw"}
    assert InputConfig().channels == ("LAP2", None, "DAPI") and InputConfig().markers == ["LAP2", "DAPI"]
    with pytest.raises(ConfigError):
        InputConfig(red=None, green=None, blue=None)


def test_resolve_on_synthetic_dataset(cfg, dataset):
    from mxdata.errors import ConfigError
    from mxseg.inputs import InputConfig, resolve_input
    out, _ = dataset
    spec = resolve_input(InputConfig(red="PanCK", green=None, blue="DAPI"), out, cfg.slides)
    n = spec.norms[cfg.slides[0]]
    assert set(n) == {"PanCK", "DAPI"} and all(n[m][0] < n[m][1] for m in n)        # the dataset's stored statistics
    with pytest.raises(ConfigError, match="no statistics"):
        resolve_input(InputConfig(red="NOPE", green=None, blue="DAPI"), out, cfg.slides)


def test_baseline_config_input_block(tmp_path):
    from mxseg.baselines import from_config
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump({"baseline": "unet", "params": {"weights": "x.pt"}, "input": {"red": "PanCK", "green": None, "blue": "DAPI"}}))
    b, name = from_config(p)
    assert name == "b" and b.input_cfg.channels == ("PanCK", None, "DAPI") and b.input is None
    shipped = {"red": "LAP2", "green": None, "blue": "DAPI"}                          # the segmenters: envelope + nuclei, nothing else
    for f in list((ROOT / "configs").glob("train_*.yaml")) + list((ROOT / "configs" / "baselines").glob("*_flows.yaml")):
        assert yaml.safe_load(f.read_text())["input"] == shipped, f.name
    ssl = yaml.safe_load((ROOT / "configs" / "kodiak_pretrain.yaml").read_text())["data"]["input"]
    assert (ssl["red"], ssl["blue"]) == ("LAP2", "DAPI") and "blank" in ssl["green"] and sum(ssl["green"].values()) > 0


def test_stack_train_and_eval_end_to_end(synth, cfg, dataset, tmp_path):
    import json

    from mxdata import save_resolved as save_data_config
    from mxseg import get_baseline, run_baseline
    from mxseg.inputs import InputConfig
    from mxseg.train import train
    from mxseg.train.config import TrainConfig, UNetTrainConfig, ValConfig
    out, _ = dataset
    data_yaml = tmp_path / "data.yaml"
    save_data_config(cfg, data_yaml)
    tcfg = TrainConfig(name="unet_stack", data=str(data_yaml), dataset_dir=str(out), model="unet",
                       input=InputConfig(red="PanCK", green=None, blue="DAPI"), unet=UNetTrainConfig(encoder_weights=None, min_area=5),
                       epochs=1, batch_size=4, accelerator="cpu", val=ValConfig(every_n_epochs=1, n_metric_patches=2),
                       out=str(tmp_path / "runs"))
    run_dir = train(tcfg)
    meta = torch.load(run_dir / "weights.pt", map_location="cpu", weights_only=True)["meta"]
    assert meta["input"]["channels"] == ["PanCK", None, "DAPI"] and cfg.slides[0] in meta["input"]["norms"]   # the recipe travels with the weights
    assert json.loads((run_dir / "data_manifest.json").read_text())["input"]["channels"] == ["PanCK", None, "DAPI"]
    b = get_baseline("unet", weights=str(run_dir / "weights.pt"), device="cpu")
    run_baseline(b, cfg, split="val", dataset_dir=out, results_dir=tmp_path / "results", n=2, progress=False, store_predictions=False)
    assert b.input_markers() == ["PanCK", "DAPI"] and b.input_fixed                  # eval reads the model's markers out of the stored stack
    assert json.loads((tmp_path / "results" / "val" / "unet_unet_stack" / "run.json").read_text())["input"]["channels"] == ["PanCK", None, "DAPI"]
