from pathlib import Path
import copy
import subprocess

import torch
import yaml

from scripts.export_production import ProductionEncoder
from spectral_lejepa.models.vit_1d import build_model, EvalBackbone

ROOT = Path(__file__).resolve().parents[1]


def test_final_recipe_preserves_validated_training_and_fixes_evaluation_interface():
    cfg = yaml.safe_load((ROOT / 'configs/final_model.yaml').read_text())
    previous = yaml.safe_load((ROOT / 'configs/paired_long.yaml').read_text())
    model = copy.deepcopy(previous['model']); model['num_patches'] = 48
    assert cfg['model'] == model
    assert cfg['training'] == previous['training']
    assert cfg['data'] == previous['data']
    assert cfg['experiment']['seed'] == 101
    evaluation = yaml.safe_load((ROOT / 'configs/eval_final_model.yaml').read_text())
    assert evaluation['evaluation']['readouts'] == ['flat']
    assert evaluation['evaluation']['flat_blocks'] == [2]
    assert evaluation['evaluation']['random_control']
    assert len(evaluation['label_sets']) == 8


def test_production_encoder_matches_block2_and_scripted_dynamic_batches(tmp_path):
    cfg = yaml.safe_load((ROOT / 'configs/final_model.yaml').read_text())
    model = build_model(cfg['model'], 245).eval()
    encoder = ProductionEncoder(model, mean=.3784, std=.5126).eval()
    compiled = torch.jit.script(encoder)
    compiled.save(str(tmp_path / 'encoder.pt'))
    loaded = torch.jit.load(str(tmp_path / 'encoder.pt'))
    for batch in (1, 3):
        raw = torch.randn(batch, 245)
        with torch.no_grad():
            expected = EvalBackbone(model)((raw - .3784) / .5126).hidden_states[2].flatten(1)
            actual = loaded(raw)
        assert actual.shape == (batch, 12288)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_launcher_dry_run_serializes_fixed_eval_and_exports():
    result = subprocess.run(['bash', 'scripts/launch_final_model.sh', '--dry-run'],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert 'configs/final_model.yaml' in result.stdout
    assert 'configs/eval_final_model.yaml' in result.stdout
    assert 'scripts.export_production' in result.stdout
    assert 'flock' in result.stdout
