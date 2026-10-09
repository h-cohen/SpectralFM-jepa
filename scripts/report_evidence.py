"""CPU-only presentation diagnostics from saved banks; no training changes.

Geometry uses a fixed final-layer mean readout. Whitening policy selection is
nested inside the same outer folds, including normalizer and probe selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from spectral_lejepa.evaluation.nested import _inner_scores, _recipe_predictions, paired_delta
from spectral_lejepa.evaluation.probe import RECIPES, r2


def spectrum(x):
    x = np.asarray(x, dtype=np.float64)
    sv = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    eig = sv ** 2 / max(len(x) - 1, 1)
    alive = eig > max(float(eig[0]) * 1e-12, 0)
    eig = eig[alive]
    p = eig / eig.sum() if eig.size else eig
    # Covariance entropy rank, distinct from trainer's singular-value entropy rank.
    rank = float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.
    cap = min(len(x) - 1, x.shape[1])
    return {'variance_fraction': p.tolist(), 'effective_rank': rank,
            'rank_fraction': rank / max(cap, 1), 'rank_cap': cap,
            'top1_variance': float(p[0]) if p.size else 0.,
            'covariance_trace': float(eig.sum())}


def geometry_diagnostic(x, seed=42):
    x = np.asarray(x, dtype=np.float64)
    tr, te = next(KFold(5, shuffle=True, random_state=seed).split(x))
    pca = PCA(n_components=min(len(tr) - 1, x.shape[1]), svd_solver='full').fit(x[tr])
    alive = pca.explained_variance_ > pca.explained_variance_[0] * 1e-12
    def project(rows):
        z = pca.transform(x[rows])[:, alive]
        return z, z / np.sqrt(pca.explained_variance_[alive])
    train_before, train_after = project(tr)
    test_before, test_after = project(te)
    def points(z):
        return z[:400, :2].tolist()
    return {'n': len(x), 'd': x.shape[1], 'n_train': len(tr), 'n_test': len(te),
            'test_rows': te.tolist(), 'before': spectrum(train_before),
            'before_test': spectrum(test_before), 'after_train': spectrum(train_after),
            'after_test': spectrum(test_after), 'points_before_train': points(train_before),
            'points_before_test': points(test_before), 'points_after_train': points(train_after),
            'points_after_test': points(test_after)}


def whitening_comparison(x, y, n_repeats=2, n_folds=5, n_inner=5, seed=42):
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float64)
    predictions = {key: np.zeros((n_repeats, len(y))) for key in ('with', 'without')}
    groups = {'with': [r for r in RECIPES if r[0].startswith('whiten')],
              'without': [r for r in RECIPES if not r[0].startswith('whiten')]}
    folds = []
    for repeat in range(n_repeats):
        for f, (tr, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed + repeat).split(x)):
            inner_seed = seed + 1000 + 10 * repeat + f
            scores = _inner_scores(x[tr], y[tr], n_inner, inner_seed)
            picked = {k: max(recipes, key=lambda rc: scores[rc]) for k, recipes in groups.items()}
            fitted = _recipe_predictions(x[tr], y[tr], x[te], picked.values(), inner_seed, 'sklearn')
            fold = {'repeat': repeat, 'fold': f, 'train_rows': tr.tolist(), 'test_rows': te.tolist()}
            for k, (norm, probe) in picked.items():
                predictions[k][repeat, te] = fitted[(norm, probe)]
                fold[k] = {'normalizer': norm, 'probe': probe, 'inner_r2': scores[(norm, probe)]}
            folds.append(fold)
    paired = paired_delta(y, predictions['with'], predictions['without'], seed=seed)
    return {'n': len(y), 'with_r2': float(np.mean([r2(y, p) for p in predictions['with']])),
            'without_r2': float(np.mean([r2(y, p) for p in predictions['without']])),
            'delta': paired['delta'], 'bootstrap_sd': paired['sd'], 'folds': folds,
            'protocol': {'n_repeats': n_repeats, 'n_folds': n_folds, 'n_inner': n_inner, 'seed': seed}}


def build_evidence(root, output, max_probe_n=5000):
    rows = []
    cache_dir = output.parent / 'evidence-cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted((root / 'outputs/long-1/eval').glob('*step497990/*/bank.npz')):
        run_id, dataset = p.parent.parent.name, p.parent.name
        seed = int(run_id.split('_s')[1][0])
        cache = cache_dir / f'layer6_v1_{run_id}_{dataset}.json'
        if cache.exists():
            saved = json.loads(cache.read_text())
            cached_probe = saved.get('whitening')
            eligible = cached_probe is not None and cached_probe['n'] <= max_probe_n
            same_source = saved.get('source_mtime_ns') == p.stat().st_mtime_ns and saved.get('source_size') == p.stat().st_size
            if same_source and (saved.get('max_probe_n') == max_probe_n or eligible):
                saved['max_probe_n'] = max_probe_n
                cache.write_text(json.dumps(saved, allow_nan=False))
                rows.append(saved)
                continue
        with np.load(p, allow_pickle=False) as bank:
            x = bank['bank__layer6'].reshape(len(bank['y']), -1)
            y = bank['y']
        # Label-independent reproducible sampling for large-set geometry only.
        idx = np.sort(np.random.default_rng(42).choice(len(x), min(2048, len(x)), replace=False))
        row = {'run': run_id, 'seed': seed, 'dataset': dataset, 'source': str(p.relative_to(root)),
               'source_size': p.stat().st_size, 'source_mtime_ns': p.stat().st_mtime_ns,
               'readout': 'layer6 mean (final encoder block, 256 dimensions)', 'checkpoint_step': 497990,
               'max_probe_n': max_probe_n,
               'geometry': geometry_diagnostic(x[idx]), 'whitening': None}
        if len(y) <= max_probe_n:
            row['whitening'] = whitening_comparison(x, y)
        else:
            row['whitening_skip'] = f'n={len(y)} exceeds CPU diagnostic limit {max_probe_n}; not evaluated'
        cache.write_text(json.dumps(row, allow_nan=False))
        rows.append(row)
        print(f"Evidence: seed {seed} {dataset} geometry + {'nested whitening' if row['whitening'] else 'geometry only'}", flush=True)
    result = {'schema_version': 1, 'rows': rows,
              'sigreg_causal_status': 'No matched SIGReg-off control measured.',
              'geometry_method': 'Covariance eigenvalue entropy rank / min(N−1,D); PCA fitted on first outer training fold. Scatter uses its first two components. Held-out rows never fit whitening.',
              'whitening_method': 'Fixed layer6 mean. Whitening-only policy versus none/standardize policy; each chooses its normalizer and RidgeCV/OLS in inner CV. Same 2×5 outer and 5-fold inner splits. Diagnostic policy comparison, not the main multi-readout scorecard.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('outputs/training-progress/story_evidence.json'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        build_evidence(args.repo_root, args.repo_root / args.output)


if __name__ == '__main__':
    main()
