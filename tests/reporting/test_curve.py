import json

from spectral_lejepa.reporting.curve import curve_table, trigger


def test_trigger_needs_five_sets():
    t = {f"s{i}": {450000: .5, 497990: .5 + (.01 if i < 5 else -.01)} for i in range(8)}
    assert trigger(t)
    t["s4"][497990] = .49
    assert not trigger(t)


def test_curve_table(tmp_path):
    ev = {}
    for m, base in (("a", .5), ("b", .7)):
        for st in (1, 2):
            d = tmp_path / f"{m}{st}" / "s"
            d.mkdir(parents=True)
            (d / "nested_results.json").write_text(json.dumps(
                {"families": {"layer2/flat": {"chosen": [{"inner_r2": base + st}, {"inner_r2": base + st}]}}}))
            ev.setdefault(m, {})[st] = tmp_path / f"{m}{st}"
    t = curve_table(ev, ["s"])
    assert t == {"s": {1: 1.6, 2: 2.6}}
