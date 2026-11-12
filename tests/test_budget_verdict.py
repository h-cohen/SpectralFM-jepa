import numpy as np

from ops.budget_verdict import blend_helps, borderline, f_star, passes, substitutes

OK = np.array([0, 0, 0, 0, -.005, -.02, -.02])          # 5/7 within .01, mean -.0064
FOUR = np.array([0, 0, 0, 0, -.02, -.02, -.02])         # 4/7 within .01
LOW_MEAN = np.array([0, 0, 0, 0, 0, -.05, -.05])        # 5/7 but mean -.0143


def test_passes_needs_five_of_seven_and_mean():
    assert passes(OK) and not passes(FOUR) and not passes(LOW_MEAN)


def test_f_star_is_smallest_passing_fraction():
    assert f_star({1.0: OK, .25: OK, .0625: FOUR, .015: FOUR}) == .25
    assert f_star({1.0: FOUR, .25: FOUR}) is None


def test_substitutes_when_ms_passes_where_m_fails():
    m = {1.0: OK, .25: FOUR, .0625: FOUR}
    assert substitutes(m, {.25: OK, .0625: FOUR})
    assert not substitutes(m, {.25: FOUR, .0625: FOUR})


def test_blend_helps_rule():
    assert blend_helps(np.array([.03, .03, .02, .01, .01, -.01, -.01]))      # 5/7 > 0, mean .0129
    assert not blend_helps(np.array([.02, .02, .01, .01, .01, -.01, -.01]))  # 5/7 but mean .0071
    assert not blend_helps(np.array([.02, .02, .02, .02, -.01, -.01, -.01]))  # 4/7


def test_borderline_flags_mean_near_bar():
    assert borderline(np.full(7, -.012)) and not borderline(np.full(7, .02))


def test_summarize_on_fake_fusion(tmp_path):
    import json
    from ops.budget_verdict import SEVEN, _labels, fig_budget, summarize
    rng = np.random.default_rng(0)
    for ds in SEVEN:
        (tmp_path / ds).mkdir()
        y = rng.standard_normal(30)
        arr = {'y': y, 'raw': y + rng.standard_normal((2, 30)), 'random_avg_a': y + 2 * rng.standard_normal((2, 30))}
        arr |= {f'avg_{lab}': y + .5 * rng.standard_normal((2, 30)) for lab in _labels()}
        np.savez(tmp_path / ds / 'fusion_oof.npz', **arr)
    (tmp_path / 'fusion.json').write_text(json.dumps({'sets': list(SEVEN), 'mix_groups': None}))
    out = summarize(tmp_path)
    assert set(out['f_star']) == {'M', 'MS', 'MX'} and out['mean_minus_a']['a'] == 0
    assert out['mean_vs_raw']['bm6'] > 0
    fig_budget(out, tmp_path / 'fig')
    assert (tmp_path / 'fig' / 'fig_budget.svg').exists()


def test_substitutes_ignores_full_fraction_reference():
    zero = np.zeros(7)
    assert not substitutes({1.0: FOUR, .25: FOUR}, {1.0: zero, .25: FOUR})
