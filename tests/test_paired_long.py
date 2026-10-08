import subprocess
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


def test_launcher_dry_run_emits_pair_and_serial_evaluation(tmp_path):
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/launch_paired_long.sh'), '--dry-run', '48', '2', '6'],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert 'model.num_patches=48' in result.stdout
    assert 'experiment.seed=2' in result.stdout
    assert 'CUDA_VISIBLE_DEVICES=6' in result.stdout
    assert 'flock' in result.stdout
    assert result.stdout.count('CUDA_VISIBLE_DEVICES=6') == 2
    assert result.stdout.count('evaluation.lock') == 1


def test_launcher_rejects_unplanned_variant():
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/launch_paired_long.sh'), '--dry-run', '12', '0', '0'],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert 'variant must be 24 or 48' in result.stderr
