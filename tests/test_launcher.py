import subprocess


def test_final_launcher_dry_run_uses_the_package_cli():
    out = subprocess.run(['bash', 'ops/launch_final_model.sh', '--dry-run'], capture_output=True, text=True, check=True).stdout
    assert 'python -m spectral_lejepa pretrain --config configs/final_model.yaml' in out
    assert 'python -m spectral_lejepa export' in out and 'python -m spectral_lejepa evaluate' in out
    assert 'scripts' not in out
