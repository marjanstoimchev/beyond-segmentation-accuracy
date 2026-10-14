import pytest

from mxdata import ConfigError, DataConfig, config_from_dict, load_config, save_resolved, to_dict


def test_defaults_are_valid():
    cfg = DataConfig()
    assert cfg.qc_policy == "ignore_wrong" and cfg.markers == ("DAPI",) and cfg.patches.overlap == 64 and cfg.supervised


def test_unknown_key_is_rejected_with_hint():
    with pytest.raises(ConfigError) as e:
        config_from_dict({"slides": ["SLIDE-0235"], "marker": ["DAPI"]})
    assert "unknown key" in str(e.value) and "markers" in str(e.value)
    with pytest.raises(ConfigError) as e:                         # the old singular key gets a pointed hint
        config_from_dict({"slide": "SLIDE-0235"})
    assert "slides:" in str(e.value)


def test_slides_and_patches():
    cfg = config_from_dict({"slides": ["A", "B"], "patches": {"size": 128, "stride": 96}})
    assert cfg.slide == "A" and cfg.with_slides("C").slides == ("C",) and cfg.patches.overlap == 32
    assert cfg.supervised and cfg.patches.effective_unlabeled_stride == 128
    assert not config_from_dict({"mode": "self_supervised"}).supervised
    with pytest.raises(ConfigError):
        config_from_dict({"patches": {"size": 64, "stride": 96}})
    with pytest.raises(ConfigError):
        config_from_dict({"slides": []})


def test_nested_unknown_key_names_the_path():
    with pytest.raises(ConfigError) as e:
        config_from_dict({"patches": {"size": 256, "strid": 128}})
    assert "config.patches" in str(e.value)


@pytest.mark.parametrize("bad", [{"level": "0"}, {"patches": {"size": -1}}, {"qc_policy": "drop_all"}, {"mode": "unsupervised"},
                                 {"patches": {"unlabeled_stride": 999}}, {"patches": {"max_unlabeled_per_slide": 0}}, {"tiles": {}},
                                 {"normalization": {"pmin": 99, "pmax": 1}}, {"markers": "DAPI"}, {"split": {"val_frac": 1.5}}])
def test_invalid_values_raise(bad):
    with pytest.raises(ConfigError):
        config_from_dict(bad)


def test_lists_become_tuples_and_roundtrip(tmp_path):
    cfg = config_from_dict({"markers": ["DAPI", "PanCK"], "regions": [1, 2], "split": {"val_regions": [2]}})
    assert cfg.markers == ("DAPI", "PanCK") and cfg.regions == (1, 2) and cfg.split.val_regions == (2,)
    per = config_from_dict({"split": {"val_regions": {"SLIDE-A": [2]}, "test_regions": {"SLIDE-A": [3], "SLIDE-B": [1]}}}).split
    assert per.val_regions == {"SLIDE-A": (2,)} and per.regions_for("test", "SLIDE-B") == {1} and per.regions_for("test", "SLIDE-C") == set()
    assert per.regions_for("val", "SLIDE-B") == set() and to_dict(per)["test_regions"] == {"SLIDE-A": [3], "SLIDE-B": [1]}
    with pytest.raises(ConfigError, match="both val_regions and test_regions"):
        config_from_dict({"split": {"val_regions": {"SLIDE-A": [2]}, "test_regions": {"SLIDE-A": [2]}}})
    p = save_resolved(cfg, tmp_path / "resolved.yaml")
    assert load_config(p) == cfg
    assert to_dict(cfg)["markers"] == ["DAPI", "PanCK"]


def test_missing_file():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/config.yaml")
