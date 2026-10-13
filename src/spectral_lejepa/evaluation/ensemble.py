"""Equal-weight ensembles of members' out-of-fold probe predictions."""
import json
from pathlib import Path

import numpy as np


def member_oof(member_dir, dataset):
    """nested_oof.npz of a pipeline job: <member>/eval/<training run>_step<N>/<dataset>/nested_oof.npz."""
    member_dir = Path(member_dir)
    run = Path(json.loads((member_dir / 'state.json').read_text())['checkpoint']).parent.name
    (bank,) = (member_dir / 'eval').glob(f'{run}_step*')
    return np.load(bank / dataset / 'nested_oof.npz')


def ensemble_predictions(oofs, arm='layer2/flat'):
    return np.mean([o[arm] for o in oofs], 0)
