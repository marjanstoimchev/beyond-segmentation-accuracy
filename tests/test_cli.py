import yaml

from mxdata.cli import main
from tests.conftest import SLIDE


def _cfg_file(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"slides": [SLIDE], "markers": ["DAPI"], "min_area_px": 20, "normalization": {"sample_pixels": 20000},
                                 "patches": {"size": 64, "stride": 48, "min_tissue": 0.2, "tissue_max_dim": 256},
                                 "split": {"val_regions": [3]}}))
    return str(p)


def test_cli_stages(synth, tmp_path, capsys):
    root, c = str(synth["root"]), _cfg_file(tmp_path)
    assert main(["report", "-c", c, "--root", root]) == 0
    assert "overlap groups" in capsys.readouterr().out
    assert main(["validate", "-c", c, "--root", root, "-v"]) == 0
    ds = tmp_path / "segdata"
    assert main(["sample", "-c", c, "--root", root, "--out", str(ds), "--dry-run"]) == 0 and not ds.exists()
    assert main(["sample", "-c", c, "--root", root, "--out", str(ds)]) == 0
    assert (ds / "patches.csv").exists() and (ds / "dataset.json").exists()
    assert any((d / "DAPI.tif").is_file() and (d / "mask.png").is_file() for d in (ds / SLIDE).glob("R*_x*_y*"))   # one folder per patch
    assert all((d / "mask_true.png").is_file() and (d / "cells.csv").is_file() for d in (ds / SLIDE).glob("R*_x*_y*"))   # confirmed mask + cell table
    assert main(["dataset", "--out", str(ds)]) == 0 and "kept" in capsys.readouterr().out
    assert main(["config", "-c", c, "--out", str(tmp_path / "resolved.yaml")]) == 0
    assert yaml.safe_load((tmp_path / "resolved.yaml").read_text())["qc_policy"] == "ignore_wrong"


def test_cli_errors_are_named(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("slide: SLIDE-9999\n")
    assert main(["report", "-c", str(bad), "--root", str(tmp_path)]) == 1 and "[E_CONFIG]" in capsys.readouterr().err
    assert main(["report", "--slides", "SLIDE-0000", "--root", str(tmp_path)]) == 1 and "[E_SLIDE]" in capsys.readouterr().err
    assert main(["dataset", "--out", str(tmp_path)]) == 1 and "[E_DATASET]" in capsys.readouterr().err
