from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_confirmatory_config_keeps_shared_training_recipe():
    config = yaml.safe_load((ROOT / 'configs/paired_long.yaml').read_text())
    assert config['model']['num_patches'] == 24
    assert config['training']['epochs'] == 10
    assert config['training']['resume'] is None
    assert config['training']['init_checkpoint'] is None
    assert config['training']['ckpt_every'] == 10000
    assert config['data']['source'] == 'packed'
    assert config['data']['source_weights'] == {'labeled_regression': 0.5, 'default': 0.5}
    assert config['wandb']['group'] == 'paired-long-confirmatory'
