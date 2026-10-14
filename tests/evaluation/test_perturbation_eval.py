from pathlib import Path

from spectral_lejepa.evaluation import perturbation_eval


def test_cli_scores_any_checkpoint_to_any_output(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(perturbation_eval, 'evaluate', lambda ck, out: seen.update(ck=ck, out=out))
    perturbation_eval.main(['--checkpoint', 'a.pt', '--output', str(tmp_path / 'p.json')])
    assert seen == {'ck': 'a.pt', 'out': Path(tmp_path / 'p.json')}
