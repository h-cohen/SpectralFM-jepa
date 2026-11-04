"""Single-encoder column next to every ensemble result of docs/TRAINING_REPORT.md.

    PYTHONPATH=src:. python ops/report_single.py [--outputs DIR] [--json outputs/single-encoder/single.json]
                                                 [--patch docs/TRAINING_REPORT.md]

The single encoder is one member of the 16, block-2 flat features, taken from its own nested-CV run
(`<set>/nested_oof.npz`, key `layer2/flat`; same 2x5 outer folds, 5 inner folds, same rows and labels as every
ensemble table). RMSE and R² are computed as in `ops/report_rmse.py` (helpers imported from it).

Choice rule (the user's): the member with the most wins vs raw, ties broken by mean ΔR². It is read off the same
outer-test scores the tables show, so its column is mildly optimistic and descriptive only, not a selection result.
The script recomputes the rule over all 16 members of `accuracy-ensemble/mix_banks.txt` and stores it.

Raw check: the member's raw OOF must equal the raw OOF of every table source (same folds, so paired deltas are
valid). `--patch` inserts or refreshes the columns in the report (it drops older single-encoder columns first).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from report_rmse import MINUS, NAME, SETS, Src, delta, f3, pm, r2m, rmse, summ  # noqa: E402

from spectral_lejepa.evaluation.errors import error_metrics  # noqa: E402
from spectral_lejepa.evaluation.nested import paired_delta  # noqa: E402

SINGLE = 'robust_s103'
SINGLE_DIR = 'signal-squeeze/members/robust_s103/eval/robust_s103_20261021-190555_step49799'
KEY = 'layer2/flat'
TAG = 'single encoder (robust_s103)'
ROOT = '/mnt5/home/hadar/nova/SpectralFM-jepa'


def oof(o, d, s):
    return np.load(Path(o) / d / s / 'nested_oof.npz')


def selection(o):
    """Wins vs raw and mean ΔR² (block-2 flat) of each of the 16 members, from outer-test R²."""
    banks = Path(f'{o}/accuracy-ensemble/mix_banks.txt').read_text().split()
    out = {}
    for b in banks:
        d = b.replace('outputs/', '', 1)
        dr = []
        for s in SETS:
            z = oof(o, d, s)
            dr.append(r2m(z['y'], z[KEY]) - r2m(z['y'], z['raw']))
        out[Path(d).parts[-3] if 'members' in d else Path(d).name] = {
            'dir': d, 'wins': int(sum(x > 0 for x in dr)), 'mean_dr2': float(np.mean(dr))}
    return out


def compute(o):
    src, res, raw_check = Src(o), {'sets': {}}, {}
    for s in SETS:
        z = oof(o, SINGLE_DIR, s)
        y, P = z['y'], z[KEY]
        mix = src.fz('accuracy-ensemble/mix', s)
        # every table source shares these folds iff its raw OOF equals the member's raw OOF
        srcs = {'main': src.main(s), 'b1': src.b1(s), 'b2': src.b2(s), 'd1': src.d1(s), 'e2': src.e2(s),
                'f': src.f(s), 'bench': src.bench(s)}
        gaps = {k: float(np.abs(np.asarray(v[1]['raw'], float) - z['raw']).max()) for k, v in srcs.items()}
        assert all(np.array_equal(v[0], y) for v in srcs.values()), s
        raw_check[NAME[s]] = {'max_abs_gap_by_source': gaps, 'raw_r2': r2m(y, z['raw'])}
        e = {'n': int(len(y)), 'single': summ(y, P), 'raw': summ(y, z['raw']),
             'vs_raw': delta(y, P, z['raw']), 'mix_minus_single': delta(y, mix['mix'], P),
             'mix_minus_single_r2_nested': paired_delta(y, mix['mix'], P)}
        e['errors'] = error_metrics(y, {'raw': z['raw'], 'single': P})['single']
        res['sets'][NAME[s]] = e
    res['raw_check'] = raw_check
    res['selection'] = selection(o)
    return res


# ---------------------------------------------------------------- markdown surgery
def cells(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def row(cs):
    return '| ' + ' | '.join(cs) + ' |'


def drop_single(block):
    """Remove columns that an earlier run inserted."""
    hdr = cells(block[0])
    keep = [j for j, c in enumerate(hdr) if 'single encoder' not in c.lower()]
    if len(keep) == len(hdr):
        return block
    out = []
    for l in block:
        c = cells(l)
        out.append(row([c[j] for j in keep]) if not l.startswith('|---') else
                   '|' + '|'.join(l.strip().strip('|').split('|')[j] for j in keep) + '|')
    return out


def find(lines, start):
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    j = i
    while j < len(lines) and lines[j].startswith('|'):
        j += 1
    return i, j


def summary_cell(label, R, mode):
    """(r2 cell, rmse cell) of the single encoder for a summary row; ('', '') when the row has no meaning."""
    S = R['sets']
    dr = [S[k]['single']['r2'] - S[k]['raw']['r2'] for k in S]
    de = [S[k]['single']['rmse'] - S[k]['raw']['rmse'] for k in S]
    n = len(S)
    L = label.lower()
    if L == 'wins vs raw':
        return f'{sum(x > 0 for x in dr)}/{n}', f'{sum(x < 0 for x in de)}/{n}'
    if L == 'lower rmse than raw':
        return '', f'{sum(x < 0 for x in de)}/{n}'
    if L == 'mean δ vs raw':
        return f3(float(np.mean(dr)), True), f3(float(np.mean(de)), True)
    if L == 'mean δ (wins)':
        return f'{f3(float(np.mean(dr)), True)} ({sum(x > 0 for x in dr)}/{n})', ''
    if L == 'worst set vs raw':
        return f3(min(dr), True), f3(max(de), True)
    if L == 'mean nrmse':
        return '', f3(float(np.mean([S[k]['single']['nrmse'] for k in S])))
    E = {k: S[k]['errors'] for k in S}
    if L == 'mean 90th-pct abs. error':
        return '', f3(float(np.mean([E[k]['p90_ae'] for k in S])))
    if L == 'mean residual slope':
        return '', f3(float(np.mean([E[k]['residual_slope'] for k in S])), nd=2)
    if L == 'per-spectrum win rate vs raw':
        return '', f3(float(np.mean([E[k]['win_rate'] for k in S])), nd=2)
    return '', ''


def insert(lines, start, after, mode, R, extra=None):
    """Insert the single-encoder column(s) after column `after`; extra = [(header, per-set fn, summary fn)]."""
    i, j = find(lines, start)
    block = drop_single(lines[i:j])
    hdr = cells(block[0])
    k = hdr.index(after) + 1
    new_h = {'both': [f'{TAG} R²', f'{TAG} RMSE'], 'r2': [TAG], 'rmse': [TAG]}[mode]
    S = R['sets']
    out = [row(hdr[:k] + new_h + hdr[k:] + [h for h, _, _ in (extra or [])])]
    al = block[1].strip().strip('|').split('|')
    out.append('|' + '|'.join(al[:k] + ['---:'] * len(new_h) + al[k:] + ['---:'] * len(extra or [])) + '|')
    for l in block[2:]:
        c = cells(l)
        lab = c[0]
        if lab in S:
            e = S[lab]
            val = {'both': [f3(e['single']['r2']), f3(e['single']['rmse'])], 'r2': [f3(e['single']['r2'])],
                   'rmse': [f3(e['single']['rmse'])]}[mode]
            ex = [fn(e) for _, fn, _ in (extra or [])]
        else:
            r2c, rc = summary_cell(lab, R, mode)
            val = {'both': [r2c, rc], 'r2': [r2c], 'rmse': [rc]}[mode]
            ex = [fn(lab, R) for _, _, fn in (extra or [])]
        out.append(row(c[:k] + val + c[k:] + ex))
    lines[i:j] = out
    return lines


def patch(path, R):
    lines = Path(path).read_text().split('\n')
    S = R['sets']
    mx = lambda e: pm(e['mix_minus_single']['d_r2'], e['mix_minus_single']['sd_r2'])
    mxe = lambda e: pm(e['mix_minus_single']['d_rmse'], e['mix_minus_single']['sd_rmse'])
    sv = lambda e: pm(e['vs_raw']['d_r2'], e['vs_raw']['sd_r2'])
    sve = lambda e: pm(e['vs_raw']['d_rmse'], e['vs_raw']['sd_rmse'])
    mean = lambda f: float(np.mean([f(S[k]) for k in S]))

    def tail(lab, R_, f):
        if lab.lower() == 'mean δ vs raw':
            return f3(mean(f), True)
        return ''

    t1_extra = [('Single encoder Δ vs raw (± SD)', sv, lambda l, R_: tail(l, R_, lambda e: e['vs_raw']['d_r2'])),
                ('Single encoder ΔRMSE vs raw (± SD)', sve, lambda l, R_: tail(l, R_, lambda e: e['vs_raw']['d_rmse'])),
                ('Family mix − single encoder, ΔR² (± SD)', mx,
                 lambda l, R_: tail(l, R_, lambda e: e['mix_minus_single']['d_r2'])),
                ('Family mix − single encoder, ΔRMSE (± SD)', mxe,
                 lambda l, R_: tail(l, R_, lambda e: e['mix_minus_single']['d_rmse']))]
    insert(lines, '| Set | n | Raw R² | Raw RMSE | Random-weight R²', 'Raw RMSE', 'both', R, t1_extra)
    insert(lines, '| Set | n | Raw R² | Raw RMSE | MiniRocket R²', 'Raw RMSE', 'both', R)
    insert(lines, '| Set | Raw R² | Raw RMSE | data2vec (Feb 25) R²', 'Raw RMSE', 'both', R)
    insert(lines, '| Set | Raw | data2vec (Feb 25) | data2vec (recent) | JEPA ens. |', 'Raw', 'rmse', R)
    insert(lines, '| Set | Raw | data2vec (Feb 25) | data2vec (recent) | JEPA ens. | MiniRocket s0 (s1', 'Raw', 'r2', R)
    insert(lines, '| Set | Raw | data2vec (Feb 25) | data2vec (recent) | JEPA ens. | MiniRocket s0 |', 'Raw', 'rmse', R)
    insert(lines, '| Set | raw R² | raw RMSE | average R²', 'raw RMSE', 'both', R)
    insert(lines, '| Set | raw R² | raw RMSE | average (4) R²', 'raw RMSE', 'both', R)
    insert(lines, '| Set | raw R² | raw RMSE | R4', 'raw RMSE', 'both', R)
    insert(lines, '| Set | raw R² | raw RMSE | MIX2 R²', 'raw RMSE', 'both', R)
    insert(lines, '| Set | raw R² | raw RMSE | random R² | random RMSE | unadapted', 'raw RMSE', 'both', R)
    insert(lines, '| Set | n | raw R² | raw RMSE', 'raw RMSE', 'both', R)
    rebold(lines)
    Path(path).write_text('\n'.join(lines))


def rebold(lines):
    """Bold the best arm of each set row (single encoder included) in the tables that bold their best arm.
    Higher R² or lower RMSE; ties at 3 dp all bold. Each spec: header start, kind ('both', 'r2', 'rmse'),
    arms that compete (header prefixes; None = every column after Set)."""
    S = '| Set | '
    sg = 'single encoder (robust_s103)'
    ap = ('Raw', sg, 'JEPA', 'MiniRocket', 'pred_avg', 'concat')
    ap = tuple(x for x in ap if x != 'Raw')
    specs = [(S + f'Raw R² | Raw RMSE | {sg} R² | {sg} RMSE | data2vec', 'both', None),
             (S + f'Raw | {sg} | data2vec (Feb 25) | data2vec (recent) | JEPA ens. |', 'rmse', None),
             (S + f'Raw | {sg} | data2vec (Feb 25) | data2vec (recent) | JEPA ens. | MiniRocket s0 (s1', 'r2', ap),
             (S + f'Raw | {sg} | data2vec (Feb 25) | data2vec (recent) | JEPA ens. | MiniRocket s0 |', 'rmse', ap),
             (S + f'raw R² | raw RMSE | {sg} R² | {sg} RMSE | average R²', 'both', (sg, 'average', 'stack', 'aug.')),
             (S + f'raw R² | raw RMSE | {sg} R² | {sg} RMSE | average (4) R²', 'both',
              (sg, 'average (4)', 'A:', 'B:', 'C:'))]
    num = lambda c: round(float(c.replace('**', '').replace(MINUS, '-').split()[0]), 3)
    for start, kind, names in specs:
        i, j = find(lines, start)
        hdr = cells(lines[i])
        elig = [k for k in range(1, len(hdr)) if names is None or hdr[k].startswith(names)]
        for l in range(i + 2, j):
            c = [x.replace('**', '') for x in cells(lines[l])]
            if c[0] not in NAME.values():
                continue
            for kk in (('r2', 'rmse') if kind == 'both' else (kind,)):
                cols = [k for k in elig if kind != 'both' or hdr[k].endswith('RMSE') == (kk == 'rmse')]
                v = {k: num(c[k]) for k in cols}
                best = (max if kk == 'r2' else min)(v.values())
                for k in cols:
                    if v[k] == best:
                        c[k] = f'**{c[k]}**'
            lines[l] = row(c)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--outputs', default=f'{ROOT}/outputs')
    ap.add_argument('--json', default=f'{ROOT}/outputs/single-encoder/single.json')
    ap.add_argument('--patch')
    ap.add_argument('--bold-only', action='store_true', help='only re-bold the best arm per row of --patch')
    a = ap.parse_args(argv)
    if a.bold_only:
        lines = Path(a.patch).read_text().split('\n')
        rebold(lines)
        Path(a.patch).write_text('\n'.join(lines))
        return
    R = compute(a.outputs)
    worst = max(g for v in R['raw_check'].values() for g in v['max_abs_gap_by_source'].values())
    print('largest |raw OOF gap| between the single encoder and any table source:', worst)
    sel = sorted(R['selection'].items(), key=lambda kv: (-kv[1]['wins'], -kv[1]['mean_dr2']))
    print('selection (wins, mean ΔR²):', [(k, v['wins'], round(v['mean_dr2'], 3)) for k, v in sel[:4]])
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(R, indent=1))
    if a.patch:
        patch(a.patch, R)


if __name__ == '__main__':
    main()
