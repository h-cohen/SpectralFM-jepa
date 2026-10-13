"""Repository root and atomic JSON output for study manifests and results."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/spectral_lejepa/utils/io.py -> checkout


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)
