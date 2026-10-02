from pathlib import Path

import pytest
import yaml

from spectral_lejepa.config import load_config, parse_value

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"training": {"learning_rate": 0.1, "name": "a", "tags": ["x"]}}))
    return path


def test_override_nested_value(cfg_file):
    assert load_config(cfg_file, ["training.learning_rate=0.5"])["training"]["learning_rate"] == 0.5


def test_scientific_notation_becomes_float(cfg_file):
    value = load_config(cfg_file, ["training.learning_rate=1e-3"])["training"]["learning_rate"]
    assert isinstance(value, float) and value == pytest.approx(1e-3)


def test_strings_lists_and_null(cfg_file):
    cfg = load_config(cfg_file, ["training.name=mlp", "training.tags=[a,b]", "training.learning_rate=null"])
    assert cfg["training"]["name"] == "mlp"
    assert cfg["training"]["tags"] == ["a", "b"]
    assert cfg["training"]["learning_rate"] is None


def test_unknown_key_is_rejected(cfg_file):
    with pytest.raises(KeyError, match="learning_rat"):
        load_config(cfg_file, ["training.learning_rat=1e-3"])


def test_unknown_section_is_rejected(cfg_file):
    with pytest.raises(KeyError, match="trainig"):
        load_config(cfg_file, ["trainig.learning_rate=1.0"])


def test_malformed_override_is_rejected(cfg_file):
    with pytest.raises(ValueError, match="key.sub=value"):
        load_config(cfg_file, ["training.learning_rate"])


def test_parse_value():
    assert parse_value("true") is True
    assert parse_value("256") == 256
    assert parse_value("5e-4") == pytest.approx(5e-4)
    assert parse_value("none") == "none"


def test_pretrain_yaml_numbers_are_numbers():
    cfg = load_config(REPO / "configs" / "pretrain.yaml")
    t = cfg["training"]
    for key in ("learning_rate", "weight_decay", "lambda_sigreg", "final_lr_ratio", "sigreg_t_max"):
        assert isinstance(t[key], float), key
    assert t["lambda_sigreg"] == 0.05
    assert t["batch_size"] == 256
    assert cfg["masking"]["mask_ratio"] == 0.5
    assert cfg["model"]["num_patches"] == 24
    assert cfg["data"]["sequence_length"] == 245
