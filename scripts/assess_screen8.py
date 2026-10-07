"""Summarize the completed screen-8 evaluation arms; this command never launches training."""
import argparse
import json
import math
import re
from pathlib import Path

from scripts.training_progress import SETS

ARMS = ('control_s0', 'control_s1', 'random50_s0', 'random50_s1', 'block75_s0', 'block75_s1')
VARIANTS = ('random50', 'block75')
HARD_SETS = ('dataset0106', 'dataset0109', 'dataset0120')
STABLE_SETS = ('dataset0055', 'dataset0112', 'dataset0113', 'dataset0114')


def _complete(summary):
    sets = summary.get('sets') if isinstance(summary, dict) else None
    return (isinstance(sets, dict) and set(sets) == set(SETS) and all(
        isinstance(sets[name], dict) and all(isinstance(sets[name].get(key), (int, float))
        and not isinstance(sets[name].get(key), bool) and math.isfinite(sets[name][key])
        for key in ('raw_r2', 'embedding_r2')) for name in SETS))


def assess(scorecards):
    """Compare complete arm scorecards and report exploratory promotion gates."""
    cards = {arm: scorecards[arm] for arm in ARMS
             if arm in scorecards and _complete(scorecards[arm])}
    missing = [arm for arm in ARMS if arm not in cards]
    arms = {}
    for arm in ARMS:
        seed = arm[-1]
        control = cards.get(f'control_s{seed}') if not arm.startswith('control_') else None
        summary = cards.get(arm)
        sets = {}
        for name in SETS:
            model = summary['sets'][name] if summary else {}
            base = control['sets'][name] if control else {}
            raw, score = model.get('raw_r2'), model.get('embedding_r2')
            control_raw, control_score = base.get('raw_r2'), base.get('embedding_r2')
            if raw is not None and control_raw is not None and not math.isclose(raw, control_raw, rel_tol=0, abs_tol=2e-5):
                raise ValueError(f'raw R² mismatch for {arm}/{name}: {raw} vs control {control_raw}')
            sets[name] = {'n': model.get('n'), 'raw_r2': raw, 'model_r2': score,
                'random_r2': (model.get('random_control') or {}).get('embedding_r2'),
                'delta_vs_raw': score - raw if score is not None and raw is not None else None,
                'control_raw_r2': control_raw, 'control_model_r2': control_score,
                'control_delta_vs_raw': control_score - control_raw if control_score is not None and control_raw is not None else None,
                'delta_vs_control': score - control_score if score is not None and control_score is not None else None,
                'paired_uncertainty': {'model_minus_raw': model.get('embedding_minus_raw'),
                    'model_vs_random': model.get('vs_random_control'), 'model_vs_control': model.get('vs_control'),
                    'control_minus_raw': base.get('embedding_minus_raw')}}
        deltas = [sets[name]['delta_vs_raw'] for name in SETS]
        gains = [sets[name]['delta_vs_control'] for name in SETS]
        arms[arm] = {'status': 'complete' if summary else 'pending', 'wins': sum(d > 0 for d in deltas) if summary else None,
            'mean_delta_vs_raw': sum(deltas) / len(deltas) if summary else None,
            'hard_set_mean_gain': sum(sets[n]['delta_vs_control'] for n in HARD_SETS) / len(HARD_SETS) if summary and control else None,
            'mean_gain_vs_control': sum(gains) / len(gains) if summary and control else None, 'sets': sets}
    decisions = {}
    for variant in VARIANTS:
        selected = [arms[f'{variant}_s{seed}'] for seed in (0, 1)]
        def mean_value(field):
            values = [row['sets'][name][field] for row in selected for name in SETS if row['sets'][name][field] is not None]
            return sum(values) / len(values) if values else None

        ready = all(row['status'] == 'complete' and row['mean_gain_vs_control'] is not None for row in selected)
        if ready:
            seed_gains = [{name: row['sets'][name]['delta_vs_control'] for name in SETS} for row in selected]
            overall = sum(sum(g.values()) for g in seed_gains) / (2 * len(SETS))
            ld_gain = sum(g['labeled_data'] for g in seed_gains) / 2
            stable = {name: sum(g[name] for g in seed_gains) / 2 for name in STABLE_SETS}
            gates = {'wins_each_seed': all(row['wins'] >= 5 for row in selected),
                'hard_set_gain_each_seed': all(row['hard_set_mean_gain'] > 0 for row in selected),
                'overall_mean_gain_positive': overall > 0, 'seed_mean_ld_gain_at_least_minus_0_01': ld_gain >= -0.01,
                'stable_sets_seed_mean_at_least_minus_0_05': all(value >= -0.05 for value in stable.values())}
            candidate = all(gates.values())
            decisions[variant] = {'status': 'candidate' if candidate else 'not_candidate', 'candidate': candidate,
                                  'gates': gates, 'overall_mean_gain': overall, 'seed_mean_ld_gain': ld_gain,
                                  'stable_set_mean_gains': stable}
        else:
            decisions[variant] = {'status': 'pending', 'candidate': None,
                'gates': {key: None for key in ('wins_each_seed', 'hard_set_gain_each_seed', 'overall_mean_gain_positive',
                    'seed_mean_ld_gain_at_least_minus_0_01', 'stable_sets_seed_mean_at_least_minus_0_05')},
                'overall_mean_gain': None, 'seed_mean_ld_gain': None,
                'stable_set_mean_gains': {name: None for name in STABLE_SETS}}
        decisions[variant].update({
            'mean_raw_r2': mean_value('raw_r2'), 'mean_model_r2': mean_value('model_r2'),
            'mean_random_r2': mean_value('random_r2'), 'mean_delta_vs_raw': mean_value('delta_vs_raw'),
            'mean_control_model_r2': mean_value('control_model_r2'),
            'mean_delta_vs_control': mean_value('delta_vs_control'),
        })
    return {'missing_arms': missing, 'arms': arms, 'decisions': decisions,
            'interpretation': 'Exploratory selection only; not a statistical-significance claim. A win remains model R² > raw R².'}


def _markdown(result):
    lines = ['# Screen 8 decision', '', result['interpretation'], '',
             f"Pending arms: {', '.join(result['missing_arms']) or 'none'}", '',
             '| Variant | Status | Mean raw R² | Mean model R² | Mean random R² | Mean Δ vs raw | Mean Δ vs control | Wins seed 0 | Wins seed 1 | LD seed mean |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def show(value):
        return '' if value is None else f'{value:.3f}'

    for variant, decision in result['decisions'].items():
        lines.append(f"| {variant} | {decision['status']} | {show(decision['mean_raw_r2'])} | {show(decision['mean_model_r2'])} | "
                     f"{show(decision['mean_random_r2'])} | {show(decision['mean_delta_vs_raw'])} | {show(decision['mean_delta_vs_control'])} | "
                     f"{result['arms'][variant + '_s0']['wins']} | {result['arms'][variant + '_s1']['wins']} | {show(decision['seed_mean_ld_gain'])} |")
    lines.extend(['', '## Gate details', ''])
    for variant, decision in result['decisions'].items():
        lines.append(f"### {variant}")
        lines.append('')
        lines.extend(f"- {gate}: {value}" for gate, value in decision['gates'].items())
        if decision['stable_set_mean_gains']:
            lines.extend(f"- {name} seed mean gain: {value}" for name, value in decision['stable_set_mean_gains'].items())
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/screen-8'))
    args = parser.parse_args(argv)
    root = args.root
    pattern = re.compile(rf"^({'|'.join(map(re.escape, ARMS))})_\d{{8}}-\d{{6}}_step15000$")
    found = {}
    for path in sorted((root / 'eval').glob('*/summary.json')):
        match = pattern.fullmatch(path.parent.name)
        if match:
            try:
                found[match.group(1)] = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
    result = assess(found)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'decision.json').write_text(json.dumps(result, indent=2))
    (root / 'decision.md').write_text(_markdown(result))
    import wandb
    run = wandb.init(project='spectralfm-lejepa', group='screen-8', job_type='decision',
                     config={'root': str(root), 'arms': list(ARMS)})
    columns = ['variant', 'status', 'candidate', 'mean_raw_r2', 'mean_model_r2', 'mean_random_r2', 'mean_delta_vs_raw',
               'mean_delta_vs_control', 'wins_s0', 'wins_s1', 'mean_gain', 'ld_seed_mean']
    rows = [[name, row['status'], row['candidate'], row['mean_raw_r2'], row['mean_model_r2'], row['mean_random_r2'],
             row['mean_delta_vs_raw'], row['mean_delta_vs_control'], result['arms'][name + '_s0']['wins'],
             result['arms'][name + '_s1']['wins'], row['overall_mean_gain'], row['seed_mean_ld_gain']]
            for name, row in result['decisions'].items()]
    run.log({'decision/table': wandb.Table(columns=columns, data=rows)})
    score_columns = ['arm', 'dataset', 'n', 'raw_r2', 'model_r2', 'random_r2', 'delta_vs_raw', 'control_raw_r2',
        'control_model_r2', 'control_delta_vs_raw', 'delta_vs_control', 'model_minus_raw_delta', 'model_minus_raw_sd',
        'model_minus_raw_p_a_better', 'model_vs_random_delta', 'model_vs_random_sd', 'model_vs_random_p_a_better',
        'model_vs_control_delta', 'model_vs_control_sd', 'control_minus_raw_delta', 'control_minus_raw_sd',
        'control_minus_raw_p_a_better']
    score_rows = []
    for arm, arm_data in result['arms'].items():
        for dataset, point in arm_data['sets'].items():
            uncertainty = point['paired_uncertainty']
            pairs = [uncertainty[key] or {} for key in ('model_minus_raw', 'model_vs_random', 'model_vs_control', 'control_minus_raw')]
            fields = (('delta', 'sd', 'p_a_better'), ('delta', 'sd', 'p_a_better'), ('delta', 'sd'), ('delta', 'sd', 'p_a_better'))
            score_rows.append([arm, dataset, point['n'], point['raw_r2'], point['model_r2'], point['random_r2'],
                point['delta_vs_raw'], point['control_raw_r2'], point['control_model_r2'], point['control_delta_vs_raw'],
                point['delta_vs_control'], *(pair.get(field) for pair, keys in zip(pairs, fields) for field in keys)])
    run.log({'decision/per_dataset': wandb.Table(columns=score_columns, data=score_rows)})
    for name, row in result['decisions'].items():
        for key, value in row['gates'].items():
            if value is not None:
                run.log({f'{name}/gate/{key}': int(value)})
        for key in ('overall_mean_gain', 'seed_mean_ld_gain'):
            if row[key] is not None:
                run.log({f'{name}/{key}': row[key]})
        for seed in (0, 1):
            arm_row = result['arms'][f'{name}_s{seed}']
            for key in ('wins', 'hard_set_mean_gain', 'mean_gain_vs_control'):
                if arm_row[key] is not None:
                    run.log({f'{name}/seed{seed}/{key}': arm_row[key]})
        for dataset, value in row['stable_set_mean_gains'].items():
            if value is not None:
                run.log({f'{name}/stable_set/{dataset}_mean_gain': value})
    run.finish()
    print((root / 'decision.md').read_text())


if __name__ == '__main__':
    main()
