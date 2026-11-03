"""RMSE next to every R² in docs/TRAINING_REPORT.md, from the out-of-fold predictions already on disk.

    PYTHONPATH=src:. python ops/report_rmse.py [--outputs DIR] [--json outputs/rmse/rmse.json]
                                              [--patch docs/TRAINING_REPORT.md]

No refitting. Each table's source arrays are the nested-CV predictions that produced its R² (the
`<run>/<set>/nested_oof.npz` of every member, or an ensemble's `fusion_oof.npz`; `[repeats, n]`). RMSE is
computed per repeat and averaged over repeats, exactly as R² is (`nested.py`), in the units of the stored
labels. Those labels are z-scored per set at the source (the dataset pickles already hold `parameter_0` with
mean 0 and sample SD 1), so "label units" are standard deviations of the set; the original physical scale is
not on disk. nRMSE = RMSE / population SD(y), comparable across sets. Paired ΔRMSE SDs use the same 1,000
bootstrap resamples (seed 0) as `nested.paired_delta`, so they pair with the R² Δ SDs in the report.

The script recomputes every R² and prints the largest gap to the value printed in the report, which proves
each table's source is the right one. `--patch` rewrites the tables of the report in place (they are found by
their header line, old or new, so it can be run again).
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from spectral_lejepa.evaluation.errors import error_metrics
from spectral_lejepa.evaluation.probe import r2

SETS = ('labeled_data', 'dataset0055', 'dataset0106', 'dataset0109', 'dataset0112', 'dataset0113',
        'dataset0114', 'dataset0120')
NAME = {s: s.replace('dataset', '') for s in SETS}
MINUS = '−'


# ---------------------------------------------------------------- metrics
def rmse(y, P):
    return float(np.mean(np.sqrt(((np.asarray(P, float) - y) ** 2).mean(1))))


def r2m(y, P):
    return float(np.mean([r2(y, p) for p in P]))


def summ(y, P):
    return {'r2': r2m(y, P), 'rmse': rmse(y, P), 'nrmse': rmse(y, P) / float(y.std())}


def delta(y, A, B, n_boot=1000, seed=0):
    """A minus B on the same folds: ΔR² and ΔRMSE with bootstrap SDs (resamples as in nested.paired_delta)."""
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    dr = np.array([np.mean([r2(y[i], a[i]) - r2(y[i], b[i]) for a, b in zip(A, B)]) for i in boots])
    de = np.array([rmse(y[i], A[:, i]) - rmse(y[i], B[:, i]) for i in boots])
    return {'d_r2': r2m(y, A) - r2m(y, B), 'sd_r2': float(dr.std()),
            'd_rmse': rmse(y, A) - rmse(y, B), 'sd_rmse': float(de.std())}


# ---------------------------------------------------------------- sources (one function per family of tables)
class Src:
    def __init__(self, out):
        self.o = Path(out)

    def fz(self, p, s):
        return np.load(self.o / p / s / 'fusion_oof.npz')

    def main(self, s):
        """Family mix (accuracy-ensemble/mix) and the robust four (ensemble-fusion/plain, = R4)."""
        m, p = self.fz('accuracy-ensemble/mix', s), self.fz('ensemble-fusion/plain', s)
        assert np.array_equal(m['y'], p['y']) and np.array_equal(m['raw'], p['raw'])
        return m['y'], {'raw': m['raw'], 'random': m['random_avg'], 'R4': p['avg'], 'mix': m['mix'],
                        'NR8': m['avg_a'], 'R8': m['avg_b'], 'M16': m['avg']}

    def b1(self, s):
        p, t = self.fz('ensemble-fusion/plain', s), self.fz('ensemble-fusion/tta-stack', s)
        return p['y'], {'raw': p['raw'], 'avg': p['avg'], 'stack': p['stack'], 'aug_avg': t['avg'],
                        'aug_stack': t['stack'], 'random': p['random_avg']}

    def b2(self, s):
        p, sel, m = self.fz('ensemble-fusion/plain', s), self.fz('signal-squeeze/select', s), \
            self.fz('accuracy-ensemble/mix', s)
        ad = [np.load(glob.glob(str(self.o / f'signal-squeeze/adapt/eval/{k}/*/{s}/nested_oof.npz'))[0])
              for k in ('p0', 'p1', 's101', 's401')]
        assert all(np.array_equal(a['y'], p['y']) for a in ad)
        return p['y'], {'raw': p['raw'], 'avg': p['avg'], 'select': sel['select'], 'eight': m['avg_b'],
                        'adapt': np.mean([a['layer2/flat'] for a in ad], 0), 'random': m['random_avg']}

    def d1(self, s):
        m = self.fz('accuracy-ensemble/mix3', s)
        return m['y'], {k: m[k] for k in ('raw', 'MIX2', 'MIX2s', 'MIX3', 'MIX3s')} | {'p24': m['avg_c']}

    def e1(self, s):
        lin, al, nl = (self.fz(f'probes-nl/{k}', s) for k in ('lin', 'all', 'nl'))
        return lin['y'], {'raw_lin': lin['raw'], 'raw_all': al['raw'], 'random': lin['random_avg'],
                          'MIX2': lin['mix'], 'MIX2all': al['mix'], 'MIX2nl': nl['mix']}

    def e2(self, s):
        a, b = self.fz('set-adapt/select/ens_perset', s), self.fz('set-adapt/select/ens_pooled', s)
        return a['y'], {'raw': a['raw'], 'random': a['random_avg'], 'unadapted': a['avg_u'],
                        'perset': a['avg_d'], 'pooled': b['avg_d']}

    def f(self, s):
        a = self.fz('synth-only/ens', s)
        return a['y'], {'raw': a['raw'], 'random': a['random_avg'], 'all': a['avg_a'], 'synth': a['avg_s']}

    def bench(self, s):
        """Section 9, section 10 and Appendix A: robust four (jepa), raw, data2vec and ROCKET-family arms."""
        from spectral_lejepa.evaluation import benchmark
        members = [*sorted(self.o.glob('robust-ensemble/robust_*/eval/*_step*')),
                   *self.o.glob('robustness-continuation/robust_s401/eval/*_step*')]
        y, P = benchmark.predictions(self.o / 'benchmark' / s, members)
        return y, P


TABLES = {  # id: (source, arms, pairs of (arm, reference) for paired deltas with SDs)
    't1': ('main', ('raw', 'random', 'R4', 'mix'), (('mix', 'raw'), ('R4', 'raw'), ('mix', 'random'))),
    'sec9': ('bench', ('raw', 'data2vec_feb25', 'data2vec', 'jepa'), (('jepa', 'raw'),)),
    'appA': ('bench', ('raw', 'data2vec_feb25', 'data2vec', 'jepa', 'minirocket_s0', 'rocket', 'multirocket',
                       'pred_avg', 'concat'), ()),
    'b1': ('b1', ('raw', 'avg', 'stack', 'aug_avg', 'aug_stack', 'random'), ()),
    'b2': ('b2', ('raw', 'avg', 'select', 'eight', 'adapt', 'random'), ()),
    'd': ('main', ('raw', 'R4', 'NR8', 'R8', 'M16', 'mix'), ()),
    'd1': ('d1', ('raw', 'MIX2', 'MIX2s', 'MIX3', 'MIX3s', 'p24'), ()),
    'e1': ('e1', ('raw_lin', 'raw_all', 'random', 'MIX2', 'MIX2all', 'MIX2nl'),
           (('MIX2all', 'MIX2'),)),
    'e2': ('e2', ('raw', 'random', 'unadapted', 'perset', 'pooled'), ()),
    'f': ('f', ('raw', 'random', 'all', 'synth'), (('synth', 'all'),)),
}


def compute(out):
    src, res = Src(out), {}
    for tid, (fn, arms, pairs) in TABLES.items():
        res[tid] = {}
        for s in SETS:
            try:
                y, P = getattr(src, fn)(s)
            except (FileNotFoundError, IndexError):
                continue                       # E1: labeled_data was never finished
            if tid == 'e1' and s == 'labeled_data':
                continue
            P = {k: np.asarray(v, float) for k, v in P.items()}
            e = {'n': int(len(y)), 'label_sd': float(y.std()), 'arms': {a: summ(y, P[a]) for a in arms},
                 'pairs': {f'{a}-{b}': delta(y, P[a], P[b]) for a, b in pairs}}
            if tid == 'sec9':
                e['errors'] = {k: v for k, v in error_metrics(y, P).items()}
            res[tid][NAME[s]] = e
        print(f'computed {tid}: {len(res[tid])} sets')
    return res


# ---------------------------------------------------------------- formatting
def f3(x, sign=False, nd=3):
    s = f'{abs(x):.{nd}f}'
    s = s[1:] if s.startswith('0.') else s
    if round(x, nd) == 0:
        return ('+' if sign else '') + s
    return (MINUS if x < 0 else ('+' if sign else '')) + s


def bold(txt, on):
    return f'**{txt}**' if on else txt


def pm(d, sd):
    return f'{f3(d, True)} ± {f3(sd).lstrip(MINUS)}'


class T:
    """Builds one per-set table: columns are (arm, R² label, RMSE label); `extra` are (header, per-set cell fn)."""

    def __init__(self, R, tid, arms, lead=('Set',), bold_arms=()):
        self.R, self.tid, self.arms, self.bold_arms = R[tid], tid, arms, bold_arms
        self.sets = list(self.R)

    def v(self, s, arm, key):
        return self.R[s]['arms'][arm][key]

    def per_set(self, s, lead_cells, extra_cells):
        cells = [NAME.get(s, s)] + lead_cells
        for arm, _, _ in self.arms:
            r, e = self.v(s, arm, 'r2'), self.v(s, arm, 'rmse')
            br = arm in self.bold_arms and round(r, 3) == max(round(self.v(s, a, 'r2'), 3) for a in self.bold_arms)
            be = arm in self.bold_arms and round(e, 3) == min(round(self.v(s, a, 'rmse'), 3) for a in self.bold_arms)
            cells += [bold(f3(r), br), bold(f3(e), be)]
        return cells + extra_cells

    def header(self, lead_hdr, extra_hdr, aligns_lead):
        h = list(lead_hdr)
        for _, a, b in self.arms:
            h += [a, b]
        h += [x for x, _ in extra_hdr]
        al = aligns_lead + ['---:'] * (2 * len(self.arms)) + ['---:' for _ in extra_hdr]
        return h, al


def md(h, al, rows):
    return ['| ' + ' | '.join(h) + ' |', '|' + '|'.join(al) + '|'] + ['| ' + ' | '.join(r) + ' |' for r in rows]


def summary_rows(R, tid, arms, ncol_lead, specs, nd=3):
    """specs: list of (label, kind, ref_arm, arms_to_show). kinds: wins, mean, worst, nrmse."""
    rows = []
    sets = list(R[tid])
    for label, kind, ref, show in specs:
        cells = [label] + [''] * (ncol_lead - 1)
        for arm, _, _ in arms:
            if arm not in show:
                cells += ['', '']
                continue
            dr = [R[tid][s]['arms'][arm]['r2'] - R[tid][s]['arms'][ref]['r2'] for s in sets] if ref else None
            de = [R[tid][s]['arms'][arm]['rmse'] - R[tid][s]['arms'][ref]['rmse'] for s in sets] if ref else None
            if kind == 'wins':
                cells += [f'{sum(x > 0 for x in dr)}/{len(sets)}', f'{sum(x < 0 for x in de)}/{len(sets)}']
            elif kind == 'mean':
                cells += [f3(float(np.mean(dr)), True, nd), f3(float(np.mean(de)), True, nd)]
            elif kind == 'worst':
                cells += [f3(min(dr), True), f3(max(de), True)]
            elif kind == 'nrmse':
                cells += ['', f3(float(np.mean([R[tid][s]['arms'][arm]['nrmse'] for s in sets])))]
        rows.append(cells)
    return rows


def build_tables(R, weights):
    out = {}
    ds = lambda tid, s: R[tid][s]
    allsets = lambda tid: list(R[tid])

    # t1: bottom line
    arms = [('raw', 'Raw R²', 'Raw RMSE'), ('random', 'Random-weight R²', 'Random-weight RMSE'),
            ('R4', 'Previous: robust ensemble R²', 'Previous: robust ensemble RMSE'),
            ('mix', 'Family-mix R²', 'Family-mix RMSE')]
    t = T(R, 't1', arms)
    rows = []
    for s in allsets('t1'):
        p = R['t1'][s]['pairs']['mix-raw']
        rows.append(t.per_set(s, [f"{R['t1'][s]['n']:,}"], [pm(p['d_r2'], p['sd_r2']), pm(p['d_rmse'], p['sd_rmse'])]))
    h, al = t.header(['Set', 'n'], [('Δ vs raw (± SD)', 0), ('ΔRMSE vs raw (± SD)', 0)], ['---', '---:'])
    sets = allsets('t1')
    mean_d = lambda k: float(np.mean([R['t1'][s]['pairs']['mix-raw'][k] for s in sets]))
    rows.append(['mean Δ vs raw', ''] + [''] * 8 + [f3(mean_d('d_r2'), True), f3(mean_d('d_rmse'), True)])
    rows += summary_rows(R, 't1', arms, 2, [('mean nRMSE', 'nrmse', None, [a for a, _, _ in arms])])
    rows[-1] += ['', '']
    out['t1'] = md(h, al, rows)

    # sec9
    arms = [('raw', 'Raw R²', 'Raw RMSE'), ('data2vec_feb25', 'data2vec (Feb 25) R²', 'data2vec (Feb 25) RMSE'),
            ('data2vec', 'data2vec (recent) R²', 'data2vec (recent) RMSE'), ('jepa', 'JEPA ens. R²', 'JEPA ens. RMSE')]
    t = T(R, 'sec9', arms, bold_arms=[a for a, _, _ in arms])
    rows = [t.per_set(s, [], []) for s in allsets('sec9')]
    h, al = t.header(['Set'], [], ['---'])
    rows += summary_rows(R, 'sec9', arms, 1, [
        ('wins vs raw', 'wins', 'raw', [a for a, _, _ in arms[1:]]),
        ('mean Δ vs raw', 'mean', 'raw', [a for a, _, _ in arms[1:]]),
        ('mean nRMSE', 'nrmse', None, [a for a, _, _ in arms])])
    out['sec9'] = md(h, al, rows)

    # b1
    arms = [('raw', 'raw R²', 'raw RMSE'), ('avg', 'average R²', 'average RMSE'), ('stack', 'stack R²', 'stack RMSE'),
            ('aug_avg', 'aug. average R²', 'aug. average RMSE'), ('aug_stack', 'aug. stack R²', 'aug. stack RMSE'),
            ('random', 'random-weight average R²', 'random-weight average RMSE')]
    t = T(R, 'b1', arms, bold_arms=['avg', 'stack', 'aug_avg', 'aug_stack'])
    rows = [t.per_set(s, [], []) for s in allsets('b1')]
    h, al = t.header(['Set'], [], ['---'])
    sh = ['avg', 'stack', 'aug_avg', 'aug_stack']
    rows += summary_rows(R, 'b1', arms, 1, [('wins vs raw', 'wins', 'raw', sh), ('mean nRMSE', 'nrmse', None,
                                                                                [a for a, _, _ in arms])])
    out['b1'] = md(h, al, rows)

    # b2
    arms = [('raw', 'raw R²', 'raw RMSE'), ('avg', 'average (4) R²', 'average (4) RMSE'),
            ('select', 'A: selection R²', 'A: selection RMSE'), ('eight', 'B: 8 members R²', 'B: 8 members RMSE'),
            ('adapt', 'C: adapted R²', 'C: adapted RMSE'), ('random', 'random-weight (8) R²', 'random-weight (8) RMSE')]
    t = T(R, 'b2', arms, bold_arms=['avg', 'select', 'eight', 'adapt'])
    rows = [t.per_set(s, [], []) for s in allsets('b2')]
    h, al = t.header(['Set'], [], ['---'])
    sh = ['select', 'eight', 'adapt']
    rows += summary_rows(R, 'b2', arms, 1, [('wins vs average', 'wins', 'avg', sh),
                                            ('mean Δ vs average', 'mean', 'avg', sh),
                                            ('mean nRMSE', 'nrmse', None, [a for a, _, _ in arms])])
    out['b2'] = md(h, al, rows)

    # d
    arms = [('raw', 'raw R²', 'raw RMSE'), ('R4', 'R4 (previous) R²', 'R4 (previous) RMSE'),
            ('NR8', 'NR8 mean R²', 'NR8 mean RMSE'), ('R8', 'R8 mean R²', 'R8 mean RMSE'),
            ('M16', 'M16 (all 16 averaged) R²', 'M16 (all 16 averaged) RMSE'),
            ('mix', 'family mix R²', 'family mix RMSE')]
    t = T(R, 'd', arms)
    rows = [t.per_set(s, [], [weights[s]]) for s in allsets('d')]
    h, al = t.header(['Set'], [('mean non-robust weight', 0)], ['---'])
    sh = [a for a, _, _ in arms[1:]]
    sr = summary_rows(R, 'd', arms, 1, [('mean Δ vs raw', 'mean', 'raw', sh), ('worst set vs raw', 'worst', 'raw', sh),
                                        ('mean nRMSE', 'nrmse', None, [a for a, _, _ in arms])])
    out['d'] = md(h, al, rows + [r + [''] for r in sr])

    # d1
    arms = [('raw', 'raw R²', 'raw RMSE'), ('MIX2', 'MIX2 R²', 'MIX2 RMSE'), ('MIX2s', 'MIX2s R²', 'MIX2s RMSE'),
            ('MIX3', 'MIX3 R²', 'MIX3 RMSE'), ('MIX3s', 'MIX3s R²', 'MIX3s RMSE'),
            ('p24', '24-patch mean R²', '24-patch mean RMSE')]
    t = T(R, 'd1', arms)
    rows = [t.per_set(s, [], []) for s in allsets('d1')]
    h, al = t.header(['Set'], [], ['---'])
    sh = ['MIX2', 'MIX2s', 'MIX3', 'MIX3s']
    rows += summary_rows(R, 'd1', arms, 1, [('mean Δ vs raw', 'mean', 'raw', sh), ('worst set vs raw', 'worst', 'raw', sh),
                                            ('wins vs MIX2', 'wins', 'MIX2', ['MIX2s', 'MIX3', 'MIX3s']),
                                            ('mean nRMSE', 'nrmse', None, [a for a, _, _ in arms])])
    out['d1'] = md(h, al, rows)

    # e1
    arms = [('raw_lin', 'raw, linear R²', 'raw, linear RMSE'), ('raw_all', 'raw, all recipes R²', 'raw, all recipes RMSE'),
            ('random', 'random R²', 'random RMSE'), ('MIX2', 'MIX2 R²', 'MIX2 RMSE'),
            ('MIX2all', 'MIX2-all R²', 'MIX2-all RMSE'), ('MIX2nl', 'MIX2-nl R²', 'MIX2-nl RMSE')]
    t = T(R, 'e1', arms)
    rows = []
    for s in allsets('e1'):
        p = R['e1'][s]['pairs']['MIX2all-MIX2']
        rows.append(t.per_set(s, [], [pm(p['d_r2'], p['sd_r2']), pm(p['d_rmse'], p['sd_rmse'])]))
    h, al = t.header(['Set'], [('MIX2-all − MIX2 (± SD)', 0), ('MIX2-all − MIX2, ΔRMSE (± SD)', 0)], ['---'])
    sets = allsets('e1')
    A = lambda s, a, k: R['e1'][s]['arms'][a][k]
    m = lambda a, b, k: float(np.mean([A(s, a, k) - A(s, b, k) for s in sets]))
    row = ['mean Δ vs matching raw (7 sets)'] + [''] * 6 + [f3(m('MIX2', 'raw_lin', 'r2'), True),
                                                           f3(m('MIX2', 'raw_lin', 'rmse'), True),
                                                           f3(m('MIX2all', 'raw_all', 'r2'), True),
                                                           f3(m('MIX2all', 'raw_all', 'rmse'), True), '', '', '', '']
    row = ['mean Δ vs matching raw (7 sets)', '', '', '', '', '', '', f3(m('MIX2', 'raw_lin', 'r2'), True),
           f3(m('MIX2', 'raw_lin', 'rmse'), True), f3(m('MIX2all', 'raw_all', 'r2'), True),
           f3(m('MIX2all', 'raw_all', 'rmse'), True), '', '', '', '']
    nr = ['mean nRMSE (7 sets)'] + sum(([''] + [f3(float(np.mean([A(s, a, 'nrmse') for s in sets])))]
                                         for a, _, _ in arms), []) + ['', '']
    out['e1'] = md(h, al, rows + [row[:len(h)] + [''] * (len(h) - len(row)), nr])

    # e2
    arms = [('raw', 'raw R²', 'raw RMSE'), ('random', 'random R²', 'random RMSE'),
            ('unadapted', 'unadapted (4) R²', 'unadapted (4) RMSE'), ('perset', 'per-set adapted R²', 'per-set adapted RMSE'),
            ('pooled', 'pooled adapted R²', 'pooled adapted RMSE')]
    t = T(R, 'e2', arms)
    rows = []
    for s in allsets('e2'):
        c = t.per_set(s, [], [])
        # step columns (chosen by inner CV, not recomputed here) go back in front of the adapted cells
        steps = weights['e2steps'][s]
        rows.append(c[:7] + [steps[0]] + c[7:9] + [steps[1]] + c[9:])
    h = ['Set', 'raw R²', 'raw RMSE', 'random R²', 'random RMSE', 'unadapted (4) R²', 'unadapted (4) RMSE',
         'per-set step', 'per-set adapted R²', 'per-set adapted RMSE', 'pooled step', 'pooled adapted R²',
         'pooled adapted RMSE']
    al = ['---'] + ['---:'] * 12
    sr = summary_rows(R, 'e2', arms, 1, [('wins vs unadapted', 'wins', 'unadapted', ['perset', 'pooled']),
                                         ('mean Δ vs unadapted', 'mean', 'unadapted', ['perset', 'pooled']),
                                         ('mean nRMSE', 'nrmse', None, [a for a, _, _ in arms])], nd=4)
    rows += [[r[0]] + r[1:7] + [''] + r[7:9] + [''] + r[9:] for r in sr]
    out['e2'] = md(h, al, rows)

    # f
    arms = [('raw', 'raw R²', 'raw RMSE'), ('random', 'random R²', 'random RMSE'),
            ('all', 'all data (4) R²', 'all data (4) RMSE'), ('synth', 'synthetic only (4) R²', 'synthetic only (4) RMSE')]
    t = T(R, 'f', arms)
    rows = []
    for s in allsets('f'):
        p = R['f'][s]['pairs']['synth-all']
        rows.append(t.per_set(s, [f"{R['f'][s]['n']:,}"],
                              [pm(p['d_r2'], p['sd_r2']), pm(p['d_rmse'], p['sd_rmse']), weights['fshare'][s]]))
    h, al = t.header(['Set', 'n'], [('synthetic − all (± SD)', 0), ('synthetic − all, ΔRMSE (± SD)', 0),
                                    ('transductive share', 0)], ['---', '---:'])
    sets = allsets('f')
    m = lambda a, b, k: float(np.mean([R['f'][s]['arms'][a][k] - R['f'][s]['arms'][b][k] for s in sets]))
    pd_ = lambda k: float(np.mean([R['f'][s]['pairs']['synth-all'][k] for s in sets]))
    rows.append(['mean Δ vs raw', '', '', '', '', '', f3(m('all', 'raw', 'r2'), True), f3(m('all', 'raw', 'rmse'), True),
                 f3(m('synth', 'raw', 'r2'), True), f3(m('synth', 'raw', 'rmse'), True), f3(pd_('d_r2'), True),
                 f3(pd_('d_rmse'), True), ''])
    rows.append(['mean nRMSE', ''] + sum((['', f3(float(np.mean([R['f'][s]['arms'][a]['nrmse'] for s in sets])))]
                                          for a, _, _ in arms), []) + ['', '', ''])
    out['f'] = md(h, al, rows)
    return out


# ---------------------------------------------------------------- report
HEADS = {'t1': '| Set | n | Raw R²', 'sec9': '| Set | Raw | data2vec (Feb 25) | data2vec (recent) | JEPA ens. |',
         'b1': '| Set | raw | average | stack', 'b2': '| Set | raw | average (4) | A: selection',
         'd': '| Set | raw | R4 (previous)', 'd1': '| Set | raw | MIX2 | MIX2s', 'e1': '| Set | raw, linear |',
         'e2': '| Set | raw | random | unadapted (4)', 'f': '| Set | n | raw | random | all data (4)'}
HEADS_NEW = {'sec9': '| Set | Raw R² | Raw RMSE | data2vec (Feb 25) R²', 'e2': '| Set | raw R² | raw RMSE | random R²',
             'b1': '| Set | raw R² | raw RMSE | average R²', 'b2': '| Set | raw R² | raw RMSE | average (4) R²',
             'd': '| Set | raw R² | raw RMSE | R4', 'd1': '| Set | raw R² | raw RMSE | MIX2 R²',
             'e1': '| Set | raw, linear R²', 'f': '| Set | n | raw R² | raw RMSE'}


def find_block(lines, tid):
    keys = (HEADS[tid], HEADS_NEW.get(tid, HEADS[tid]))
    i = next((k for k, l in enumerate(lines) if l.startswith(keys[0]) or l.startswith(keys[1])), None)
    if i is None:
        raise SystemExit(f'table {tid}: header not found')
    j = i
    while j < len(lines) and lines[j].startswith('|'):
        j += 1
    return i, j


def old_column(lines, tid, name):
    """{set: cell} of one column of the report's current table (text kept as printed)."""
    i, j = find_block(lines, tid)
    cells = lambda l: [c.strip() for c in l.strip().strip('|').split('|')]
    k = cells(lines[i]).index(name)
    return {r[0]: r[k] for r in map(cells, lines[i + 2:j]) if r[0] in NAME.values()}


def patch(path, tables):
    lines = Path(path).read_text().split('\n')
    for tid, new in tables.items():
        i, j = find_block(lines, tid)
        lines[i:j] = new
        print(f'patched {tid}: {j - i} -> {len(new)} lines')
    Path(path).write_text('\n'.join(lines))


# ---------------------------------------------------------------- checks against the report
REPORTED = {  # R² printed in the report (3 dp), per set in SETS order; None where a column has no value
    't1': {'raw': [.830, .591, .697, .918, .377, .548, .983, .380], 'random': [.949, .541, .702, .933, .364, .561, .982, .354],
           'R4': [.959, .640, .714, .943, .421, .702, .986, .405], 'mix': [.962, .596, .706, .942, .715, .739, .985, .439]},
    'sec9': {'raw': [.830, .591, .697, .918, .377, .548, .983, .380], 'data2vec_feb25': [.853, .199, .250, .066, .233, .259, .981, .140],
             'data2vec': [.805, .336, .294, .307, .274, .541, .981, .215], 'jepa': [.959, .640, .714, .943, .421, .702, .986, .405]},
    'appA': {'minirocket_s0': [.634, .699, .719, .903, .679, .468, .986, .336], 'rocket': [.530, .605, .543, .809, .429, .521, .985, .362],
             'multirocket': [.780, .695, .668, .844, .611, .527, .985, .422], 'pred_avg': [.878, .704, .739, .937, .609, .642, .987, .427],
             'concat': [.731, .690, .712, .938, .649, .558, .987, .333]},
    'b1': {'avg': [.959, .640, .714, .943, .421, .702, .986, .405], 'stack': [.958, .691, .704, .947, .477, .688, .986, .404],
           'aug_avg': [.261, .694, .683, .824, .328, .671, .981, .311], 'aug_stack': [.266, .482, .705, .829, .401, .586, .980, .274],
           'random': [.949, .541, .702, .933, .364, .561, .983, .354]},
    'b2': {'avg': [.959, .640, .714, .943, .421, .702, .986, .405], 'select': [.960, .543, .728, .943, .438, .672, .986, .393],
           'eight': [.960, .632, .713, .942, .489, .698, .986, .411], 'adapt': [.960, .646, .709, .950, .399, .727, .986, .385],
           'random': [.949, .541, .702, .933, .364, .561, .982, .354]},
    'd': {'R4': [.959, .640, .714, .943, .421, .702, .986, .405], 'NR8': [.963, .640, .604, .926, .723, .747, .984, .446],
          'R8': [.960, .632, .713, .942, .489, .698, .986, .411], 'M16': [.963, .656, .678, .942, .653, .732, .985, .451],
          'mix': [.962, .596, .706, .942, .715, .739, .985, .439]},
    'd1': {'MIX2': [.962, .596, .706, .942, .715, .739, .985, .439], 'MIX2s': [.963, .656, .679, .942, .656, .732, .985, .451],
           'MIX3': [.963, .605, .698, .941, .714, .723, .985, .440], 'MIX3s': [.964, .671, .672, .942, .698, .734, .986, .434],
           'p24': [.963, .652, .637, .898, .721, .674, .985, .401]},
    'e1': {'raw_lin': [None, .591, .697, .918, .377, .548, .983, .380], 'raw_all': [None, .661, .708, .922, .493, .560, .984, .385],
           'random': [None, .541, .702, .933, .364, .561, .982, .354], 'MIX2': [None, .596, .706, .942, .715, .739, .985, .439],
           'MIX2all': [None, .666, .703, .939, .698, .738, .985, .464], 'MIX2nl': [None, .740, .701, .941, .681, .699, .986, .464]},
    'e2': {'raw': [.830, .591, .697, .918, .377, .548, .983, .380], 'random': [.949, .541, .702, .933, .364, .561, .982, .354],
           'unadapted': [.963, .642, .602, .924, .724, .749, .984, .448], 'perset': [.963, .653, .601, .922, .718, .745, .984, .447],
           'pooled': [.963, .661, .598, .926, .733, .746, .984, .443]},
    'f': {'raw': [.830, .591, .697, .918, .377, .548, .983, .380], 'random': [.949, .541, .702, .933, .364, .561, .982, .354],
          'all': [.963, .642, .602, .924, .724, .749, .984, .448], 'synth': [.968, .656, .617, .850, .421, .587, .984, .393]},
}
# printed values that are not computed here: random 0114 in appendix B1 reads .983 in the report; E1 prints MIX2 at 0114 etc.


APPA_RMSE = {'raw': [.412, .625, .546, .285, .781, .667, .130, .784], 'data2vec_feb25': [.383, .877, .859, .961, .869, .853, .136, .924],
             'data2vec': [.442, .798, .833, .828, .846, .672, .138, .883], 'jepa': [.203, .588, .531, .237, .752, .542, .120, .766],
             'minirocket_s0': [.605, .537, .526, .309, .562, .724, .117, .812], 'rocket': [.685, .615, .669, .434, .750, .687, .122, .794],
             'multirocket': [.469, .541, .572, .393, .619, .683, .120, .757], 'pred_avg': [.350, .533, .507, .250, .621, .594, .113, .753],
             'concat': [.518, .545, .533, .247, .587, .660, .115, .813]}   # section 10 and Appendix A RMSE tables


def check(R):
    worst = {}
    for tid, cols in REPORTED.items():
        for arm, vals in cols.items():
            for s, v in zip(SETS, vals):
                e = R[tid].get(NAME[s])
                if v is None or e is None:
                    continue
                gap = abs(round(e['arms'][arm]['r2'], 3) - v)
                if gap > 1e-9:
                    worst[(tid, arm, NAME[s])] = (round(e['arms'][arm]['r2'], 3), v)
    for arm, vals in APPA_RMSE.items():      # the two existing RMSE tables are in the same units already
        for s, v in zip(SETS, vals):
            if round(R['appA'][NAME[s]]['arms'][arm]['rmse'], 3) != v:
                worst[('rmse', arm, NAME[s])] = (round(R['appA'][NAME[s]]['arms'][arm]['rmse'], 3), v)
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--outputs', default='/mnt5/home/hadar/nova/SpectralFM-jepa/outputs')
    ap.add_argument('--json', default='/mnt5/home/hadar/nova/SpectralFM-jepa/outputs/rmse/rmse.json')
    ap.add_argument('--patch', help='report to rewrite in place')
    a = ap.parse_args(argv)
    R = compute(a.outputs)
    bad = check(R)
    print('R² mismatches vs the report (computed, printed):', bad or 'none')
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(R, indent=1))
    if a.patch:
        lines = Path(a.patch).read_text().split('\n')
        keep = {'e2steps': {}}
        for tid, col, slot in (('d', 'mean non-robust weight', None), ('f', 'transductive share', None),
                               ('e2', 'per-set step', 0), ('e2', 'pooled step', 1)):
            for s, cell in old_column(lines, tid, col).items():
                if tid == 'd':
                    keep[s] = cell
                elif tid == 'f':
                    keep.setdefault('fshare', {})[s] = cell
                else:
                    keep['e2steps'].setdefault(s, [None, None])[slot] = cell
        patch(a.patch, build_tables(R, keep))


if __name__ == '__main__':
    main()
