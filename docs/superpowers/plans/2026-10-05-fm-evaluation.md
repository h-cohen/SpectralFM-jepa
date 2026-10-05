# Foundation-Model Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score every model by whether its frozen embedding beats raw input by at least +0.05 R² on each labeled set. Add position-preserving readouts, a random-init control and a scorecard. W&B gets results and plots only.

**Architecture:**
- The bank stores each (block × readout) as one arm, and the unchanged nested CV picks among them.
- `scripts/evaluate.py` adds a random-init control pass, per-set verdicts, a scorecard table and plot in W&B, and lineage without artifacts.
- The trainer stops uploading checkpoints by default.
- The screen launcher gets its reviewed robustness fixes and no artifact.

**Tech Stack:** existing repo (Python 3.10, uv, torch 2.8, numpy, scikit-learn, matplotlib, W&B).

**Spec:** `docs/superpowers/specs/2026-10-05-fm-evaluation-design.md`

## Global Constraints

- Evaluation methodology is unchanged: same nested CV, recipes, seeds and folds. Readouts only add arms.
- With `readouts=[mean]`, bank contents and arm names must be identical to today's. The existing equivalence tests and slow tests must keep passing.
- Arm names: `layer{i}` (mean), `layer{i}/seg4`, `layer{i}/flat`. Bank keys encode `/` as `__`, e.g. `bank__layer1__seg4`.
- seg4 uses the parent's edges: `e = np.linspace(0, T, 5).astype(int)`; segment s = tokens `e[s]:max(e[s]+1, e[s+1])`, mean over tokens.
- `WIN_MARGIN = 0.05`. The verdict is `ceiling` if `1 - raw_r2 < WIN_MARGIN`, else `win` if `emb_r2 - raw_r2 >= WIN_MARGIN`, else `no-win`.
- W&B: no artifacts anywhere (evaluation, screen, checkpoints by default). Only metrics, tables and figures.
- Tests run with `WANDB_MODE=disabled`. Use `uv run` with `export PATH="$HOME/.local/bin:$PATH"` from the repo root.
- Commit trailer (as a second `-m`):
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt
  ```

## Review Focus

1. **A parent bank asked for `flat`.** It must silently provide only `mean` and `seg4`, not crash, because the parent never stored tokens. Test in Task 1.
2. **A ceiling set** (raw at 0.983) must get verdict `ceiling` and be excluded from `eligible`, never counted as a loss. Test in Task 2.
3. **A checkpoint whose W&B artifacts were deleted** must still evaluate. Lineage must not call `use_artifact`. Test in Task 2.
4. **`wandb.log_checkpoints` missing from an old config** (e.g. a checkpoint's saved config) must default to false, not raise a KeyError. Test in Task 3.
5. **A crashing arm summary in the screen launcher** must become a failed row, not lose the whole results table. Test in Task 3 (the stashed fix).

---

### Task 1: Readout banks

**Files:**
- Modify: `src/spectral_lejepa/evaluation/bank.py`, `scripts/evaluate.py` (the import and call only), `tests/test_eval_data.py`

**Interfaces:**
- Produces:
  - `READOUTS = ("mean", "seg4", "flat")`
  - `readout_arms(hidden_states: list[Tensor [B,T,D]], readouts) -> {arm_name: np.ndarray [B, d]}`
  - `extract_bank(model, signals_z, device="cuda", batch_size=64, readouts=("mean",)) -> {arm_name: [N, d]}`
  - `save_bank(path, bank, input_raw, y, meta)`, unchanged signature
  - `load_bank(path, readouts=("mean",)) -> (bank, input_raw, y, meta)`
- Removes: `extract_mean_bank`. Its only caller is `scripts/evaluate.py`, which switches to `extract_bank(..., readouts=("mean",))` in this task. Task 2 then passes the configured readouts.

- [ ] **Step 1: Write the failing tests.** In `tests/test_eval_data.py`, change the import to `from spectral_lejepa.evaluation.bank import extract_bank, load_bank, readout_arms, save_bank`. Replace `extract_mean_bank(` with `extract_bank(` in `test_bank_roundtrip_and_extraction`. That test keeps checking the mean path. Then append:

```python
def test_readout_arms_shapes_and_values():
    h = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)   # B=2, T=6, D=3
    arms = readout_arms([h, h + 1], ("mean", "seg4", "flat"))
    assert list(arms) == ["layer0", "layer0/seg4", "layer0/flat", "layer1", "layer1/seg4", "layer1/flat"]
    np.testing.assert_allclose(arms["layer0"], h.mean(1).numpy())
    e = np.linspace(0, 6, 5).astype(int)          # [0, 1, 3, 4, 6]
    seg = np.concatenate([h[:, e[s]:max(e[s] + 1, e[s + 1])].mean(1).numpy() for s in range(4)], 1)
    np.testing.assert_allclose(arms["layer0/seg4"], seg)
    assert arms["layer0/seg4"].shape == (2, 12) and arms["layer0/flat"].shape == (2, 18)
    np.testing.assert_allclose(arms["layer1/flat"], (h + 1).reshape(2, -1).numpy())


def test_bank_roundtrip_with_readouts(tmp_path):
    torch.manual_seed(0)
    backbone = EvalBackbone(build_model(SMALL, 245))
    z = normalize_like_fairseq(np.random.default_rng(0).normal(size=(10, 245)))
    bank = extract_bank(backbone, z, device="cpu", batch_size=4, readouts=READOUTS)
    assert len(bank) == 3 * (SMALL["depth"] + 1) and bank["layer2/flat"].shape == (10, 24 * 32)
    raw, y = np.zeros((10, 245), np.float32), np.arange(10.0)
    save_bank(tmp_path / "bank.npz", bank, raw, y, {"readouts": READOUTS})
    assert "bank__layer1__seg4" in np.load(tmp_path / "bank.npz").files
    back, *_ = load_bank(tmp_path / "bank.npz", readouts=READOUTS)
    assert list(back) == list(bank) and all(np.array_equal(back[k], bank[k]) for k in bank)
    only_mean, *_ = load_bank(tmp_path / "bank.npz")
    assert list(only_mean) == ["layer0", "layer1", "layer2"]


def test_parent_bank_gives_mean_and_seg4_never_flat(tmp_path):
    stats = ("mean", "std", "max", "min", "first", "last", "seg0", "seg1", "seg2", "seg3")
    arr = np.random.default_rng(0).normal(size=(6, 1, 10, 4)).astype(np.float32)
    np.savez(tmp_path / "bank.npz", bank__layer0=arr, input_raw=np.ones((6, 1, 245), np.float32),
             input_z=np.ones((6, 1, 245), np.float32), y=np.arange(6.0),
             _meta=np.array([repr({"comps": (0,), "pool_stats": stats})]))
    bank, *_ = load_bank(tmp_path / "bank.npz", readouts=READOUTS)
    assert list(bank) == ["layer0", "layer0/seg4"]
    np.testing.assert_array_equal(bank["layer0/seg4"], arr[:, 0, 6:10, :].reshape(6, -1))
```

Also add `READOUTS` to the import from `spectral_lejepa.evaluation.bank`.

- [ ] **Step 2: Run the tests to verify they fail.**
Run: `uv run pytest tests/test_eval_data.py -v`
Expected: an ImportError for `extract_bank`.

- [ ] **Step 3: Implement in `src/spectral_lejepa/evaluation/bank.py`.**
  - Replace the module docstring's first paragraph: "Representation bank: one vector per (encoder stage × readout) per spectrum." Then describe the three readouts and the key encoding.
  - Replace `extract_mean_bank` with:

```python
READOUTS = ("mean", "seg4", "flat")


def _seg4(h):
    """Means of 4 equal token segments, concatenated: [B, T, D] -> [B, 4D] (parent's edges)."""
    e = np.linspace(0, h.shape[1], 5).astype(int)
    return torch.cat([h[:, e[s]:max(e[s] + 1, e[s + 1])].mean(dim=1) for s in range(4)], dim=1)


def readout_arms(hidden_states, readouts) -> dict:
    """{arm: [B, d]}: `layer{i}` = mean over tokens, `layer{i}/seg4`, `layer{i}/flat` = all tokens."""
    fns = {"mean": lambda h: h.mean(dim=1), "seg4": _seg4, "flat": lambda h: h.reshape(len(h), -1)}
    out = {}
    for i, h in enumerate(hidden_states):
        for r in readouts:
            out[f"layer{i}" if r == "mean" else f"layer{i}/{r}"] = fns[r](h).float().cpu().numpy()
    return out


@torch.no_grad()
def extract_bank(model, signals_z, device="cuda", batch_size=64, readouts=("mean",)) -> dict:
    model.eval()
    model.to(device)
    t = torch.from_numpy(np.asarray(signals_z, dtype=np.float32))
    out = None
    for i in range(0, len(t), batch_size):
        hidden = model(input_values=t[i:i + batch_size].to(device), output_hidden_states=True).hidden_states
        arms = readout_arms(hidden, readouts)
        if out is None:
            out = {k: [] for k in arms}
        if list(arms) != list(out):
            raise RuntimeError("hidden_states changed mid-run")
        for k, v in arms.items():
            out[k].append(v)
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
```

  - In `save_bank`, encode the names: `payload = {f"bank__{s.replace('/', '__')}": ...}`.
  - Replace `load_bank` with:

```python
def load_bank(path, readouts=("mean",)):
    """(bank {arm: float32 [N, d]} in file order, input_raw [N, 245], y [N], meta).
    Our banks hold the arms they were extracted with; the parent's 10-statistic banks give
    `layer{i}` (mean) and `layer{i}/seg4` (seg0..seg3) — never `flat`."""
    data = np.load(path, allow_pickle=False)
    meta = ast.literal_eval(str(data["_meta"][0]))
    if tuple(meta.get("comps", (0,)))[0] != 0:
        raise ValueError(f"{path}: first stored component is {meta['comps'][0]}, expected 0")
    stats = tuple(meta["pool_stats"])
    bank = {}
    for key in (k for k in data.files if k.startswith("bank__")):
        name = key[len("bank__"):].replace("__", "/")
        readout = name.split("/")[1] if "/" in name else "mean"
        arr = data[key][:, 0]                                  # [N, S, D]
        if readout in readouts:
            bank[name] = np.ascontiguousarray(arr[:, stats.index("mean")] if readout == "mean"
                                              else arr[:, 0], dtype=np.float32)
        if len(stats) > 1 and "seg4" in readouts:              # parent bank: derive seg4 from seg0..seg3
            seg = [stats.index(f"seg{s}") for s in range(4)]
            bank[f"{name}/seg4"] = np.ascontiguousarray(arr[:, seg].reshape(len(arr), -1), dtype=np.float32)
    input_raw = np.ascontiguousarray(data["input_raw"][:, 0, :], dtype=np.float32)
    return bank, input_raw, np.asarray(data["y"], dtype=np.float64), meta
```

  - In `scripts/evaluate.py`, change the import `extract_mean_bank` → `extract_bank` and the call `extract_mean_bank(backbone, ...)` → `extract_bank(backbone, ..., readouts=("mean",))`. Task 2 makes it configurable.

- [ ] **Step 4: Run the tests.**
Run: `uv run pytest tests/test_eval_data.py tests/test_nested.py tests/test_evaluate_script.py -v`, then the slow equivalence and parity tests: `uv run pytest tests/test_eval_equivalence.py tests/test_eval_data.py -m slow -v` (about 5 minutes; they call `load_bank` with the default `("mean",)`), then `uv run pytest`.
Expected: all PASS.

- [ ] **Step 5: Commit.** `git add src/spectral_lejepa/evaluation/bank.py scripts/evaluate.py tests/test_eval_data.py`, then `git commit -m "Add mean/seg4/flat readout arms to the representation bank"` plus the trailer.

---

### Task 2: Evaluation: readouts, random-init control, scorecard, artifact-free W&B

**Files:**
- Modify: `scripts/evaluate.py`, `configs/eval.yaml`, `configs/eval_screen.yaml`, `tests/test_evaluate_script.py`

**Interfaces:**
- Consumes (Task 1): `extract_bank(..., readouts)`.
- Produces:
  - `WIN_MARGIN = 0.05`
  - `verdict(raw_r2, emb_r2, margin=WIN_MARGIN) -> (delta, "win" | "no-win" | "ceiling")`
  - `scorecard(sets: dict, key: str) -> {"wins", "eligible", "ceiling", "mean_delta"}`
  - Per-set `summary.json` gains `delta_vs_raw`, `verdict` and (when enabled) `random_control: {embedding_r2, best_block, delta_vs_raw, verdict}`
  - The top-level `summary.json` gains `scorecard: {model: ..., random_control: ...}`
  - Config keys `evaluation.readouts` (list) and `evaluation.random_control` (bool)

- [ ] **Step 1: Write the failing tests.** In `tests/test_evaluate_script.py`, add to the test's `ecfg["evaluation"]`: `"readouts": ["mean", "seg4", "flat"], "random_control": True`. After the existing asserts, add:

```python
    assert "pretraining_artifact" not in lin and "pretraining_artifact_digest" not in lin
    assert lin["pretraining_checkpoint"] == str(ckpt)
    assert any(k.endswith("/flat") for k in s["blocks"]) and any(k.endswith("/seg4") for k in s["blocks"])
    assert s["verdict"] in ("win", "no-win", "ceiling") and s["delta_vs_raw"] == s["embedding_r2"] - s["raw_r2"]
    rc = s["random_control"]
    assert set(rc) >= {"embedding_r2", "best_block", "delta_vs_raw", "verdict"}
    assert (out_root / SET / "random_control" / "nested_results.json").exists()
    card = top["scorecard"]
    for row in ("model", "random_control"):
        assert set(card[row]) == {"wins", "eligible", "ceiling", "mean_delta"}
        assert card[row]["eligible"] + card[row]["ceiling"] == 1
```

Add a unit test in the same file:

```python
def test_verdict_and_scorecard():
    assert evaluate.verdict(0.80, 0.86) == (pytest.approx(0.06), "win")
    assert evaluate.verdict(0.80, 0.84)[1] == "no-win"
    assert evaluate.verdict(0.983, 0.99)[1] == "ceiling"          # 1 - raw < 0.05: a +0.05 win is impossible
    sets = {"a": {"delta_vs_raw": 0.06, "verdict": "win"}, "b": {"delta_vs_raw": -0.1, "verdict": "no-win"},
            "c": {"delta_vs_raw": 0.0, "verdict": "ceiling"}}
    assert evaluate.scorecard(sets, None) == {"wins": 1, "eligible": 2, "ceiling": 1, "mean_delta": pytest.approx(-0.04 / 3)}
```

- [ ] **Step 2: Run the tests to verify they fail.**
Run: `uv run pytest tests/test_evaluate_script.py -v`
Expected: FAIL. `verdict` is missing, and so are the scorecard and random-control keys.

- [ ] **Step 3: Implement in `scripts/evaluate.py`.**
  - Docstring: drop the `wandb:` usage line. Add one line: "Readouts and the random-init control come from `evaluation.readouts` and `evaluation.random_control`; W&B gets metrics, a scorecard table and a Δ plot, and never artifacts."
  - Delete `artifact_name` and `resolve_checkpoint`. In `main`, set `ckpt_path = a.checkpoint` and delete the `use_artifact` block. Remove `pretraining_artifact` and `pretraining_artifact_digest` from `lineage`. The `--checkpoint` help becomes `"local .pt checkpoint path"`.
  - Add these helpers below `summarize`:

```python
WIN_MARGIN = 0.05


def verdict(raw_r2, emb_r2, margin=WIN_MARGIN):
    """(delta, verdict): 'ceiling' when raw leaves less than `margin` of R² to gain."""
    delta = emb_r2 - raw_r2
    if 1 - raw_r2 < margin:
        return delta, "ceiling"
    return delta, "win" if delta >= margin else "no-win"


def scorecard(sets, key):
    """Counts over per-set summaries; key=None scores the model, key='random_control' the control."""
    rows = [s if key is None else s[key] for s in sets.values()]
    eligible = [r for r in rows if r["verdict"] != "ceiling"]
    return {"wins": sum(r["verdict"] == "win" for r in eligible), "eligible": len(eligible),
            "ceiling": len(rows) - len(eligible),
            "mean_delta": float(np.mean([r["delta_vs_raw"] for r in rows])) if rows else float("nan")}


def scorecard_figure(sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(sets)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(names), 3.5))
    ax.bar(x - 0.2, [sets[n]["delta_vs_raw"] for n in names], 0.4, label="model")
    if all("random_control" in sets[n] for n in names):
        ax.bar(x + 0.2, [sets[n]["random_control"]["delta_vs_raw"] for n in names], 0.4, label="random init")
    ax.axhline(WIN_MARGIN, color="k", ls="--", lw=1, label=f"win margin +{WIN_MARGIN}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("R² − raw R² (nested CV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
```

  - Pass the configured readouts: `extract_bank(backbone, normalize_like_fairseq(raw), device=vcfg["device"], batch_size=vcfg["batch_size"], readouts=tuple(vcfg["readouts"]))`. Add `"readouts": tuple(vcfg["readouts"])` to the bank `meta` dict.
  - After `summary = summarize(nested, block)`, add `summary["delta_vs_raw"], summary["verdict"] = verdict(summary["raw_r2"], summary["embedding_r2"])`.
  - Random control: before the set loop, add
    ```python
    random_backbone = None
    if vcfg["random_control"]:
        torch.manual_seed(0)
        random_backbone = EvalBackbone(build_model(ckpt["config"]["model"], ckpt["config"]["data"]["sequence_length"]))
    ```
    Add `import torch` and `from spectral_lejepa.models.vit_1d import EvalBackbone, build_model`.
    Inside the loop, after the model's summary, add:
    ```python
    if random_backbone is not None:
        rc_dir = out_dir / "random_control"
        rc_dir.mkdir(exist_ok=True)
        rc_bank = extract_bank(random_backbone, normalize_like_fairseq(raw), device=vcfg["device"],
                               batch_size=vcfg["batch_size"], readouts=tuple(vcfg["readouts"]))
        rc_nested, rc_oof = run_nested(rc_bank, raw, y, seed=seed, n_jobs=vcfg["n_jobs"])
        write_json(rc_dir / "nested_results.json", {**rc_nested, "meta": {**lineage, "model": "random_init_seed0"}})
        np.savez_compressed(rc_dir / "nested_oof.npz", y=y, **rc_oof)
        rc_r2 = rc_nested["families"]["embedding"]["r2_mean"]
        rc_delta, rc_verdict = verdict(rc_nested["families"]["raw"]["r2_mean"], rc_r2)
        summary["random_control"] = {"embedding_r2": rc_r2, "best_block": best_block(rc_nested),
                                     "delta_vs_raw": rc_delta, "verdict": rc_verdict}
    ```
  - Extend the print line with `f", Δraw {summary['delta_vs_raw']:+.4f} [{summary['verdict']}]"`.
  - After the loop, replace the artifact block with:

```python
    card = {"model": scorecard(sets, None)}
    if random_backbone is not None:
        card["random_control"] = scorecard(sets, "random_control")
    write_json(out_root / "summary.json", {"lineage": lineage, "scorecard": card, "sets": sets})
    print("[evaluate] scorecard:", card)
    if run is not None:
        import wandb
        cols = ["set", "n", "raw_r2", "model_r2", "model_delta", "model_verdict", "random_r2", "random_delta",
                "random_verdict"]
        data = [[n, s["n"], s["raw_r2"], s["embedding_r2"], s["delta_vs_raw"], s["verdict"],
                 s.get("random_control", {}).get("embedding_r2"), s.get("random_control", {}).get("delta_vs_raw"),
                 s.get("random_control", {}).get("verdict")] for n, s in sets.items()]
        run.log({"scorecard/table": wandb.Table(columns=cols, data=data)})
        wb.log(run, flat_metrics("scorecard", card))
        wb.log_figure(run, "scorecard/delta_vs_raw", scorecard_figure(sets))
    wb.finish(run)
```

- [ ] **Step 4: Update the configs.**
  - In `configs/eval.yaml`, under `evaluation:`, add `readouts: [mean, seg4, flat]` and `random_control: true`.
  - In `configs/eval_screen.yaml`, add `readouts: [mean, seg4, flat]` and `random_control: false`. The screen compares trained arms against each other; the random control is a separate one-off.

- [ ] **Step 5: Run the tests.**
Run: `uv run pytest tests/test_evaluate_script.py -v`, then `uv run pytest`.
Expected: all PASS.

- [ ] **Step 6: Commit.** `git add scripts/evaluate.py configs/eval.yaml configs/eval_screen.yaml tests/test_evaluate_script.py`, then `git commit -m "Evaluate readout families with a random-init control and a beats-raw scorecard; no W&B artifacts"` plus the trailer.

---

### Task 3: No checkpoint uploads; screen launcher fixes

**Files:**
- Modify: `src/spectral_lejepa/training/trainer.py`, `configs/pretrain.yaml`, `scripts/screen.py`, `tests/test_train_smoke.py`, `tests/test_screen.py`

**Interfaces:**
- Produces:
  - Config key `wandb.log_checkpoints` (default false).
  - From the screen launcher: `safe_row`, `status_from_disk`, `--summarize_only`, `FOOTNOTE`, and a `stage` column.
  - The screen launcher no longer creates a W&B artifact.

- [ ] **Step 1: Restore the stashed screen fixes.** Run `git stash list`; there must be exactly one entry, "unreviewed screen.py fix-wave (agent hit usage limit)". Then `git stash pop`. It touches only `scripts/screen.py` and `tests/test_screen.py`, adding `safe_row`, `status_from_disk`, `--summarize_only`, the `control_s1` paired Δ, sign-on-Δ-only formatting, the `stage` column and `FOOTNOTE`, plus three tests. Read the diff and make sure it matches that description.

- [ ] **Step 2: Remove the screen artifact.** In `scripts/screen.py` `main`, delete the four lines that build and log `wandb.Artifact(f"{scfg['name']}-results", type="screen")`. Keep the `wandb.Table` and the summary metrics. In the module docstring, change "logged to W&B" to "logged to W&B (table and summary metrics; no artifacts)".

- [ ] **Step 3: Write the failing trainer test.** Append to `tests/test_train_smoke.py`:

```python
class FakeRun:
    id = "fake"

    def log(self, *a, **k):
        pass

    def alert(self, *a, **k):
        pass

    def finish(self):
        pass


@pytest.mark.parametrize("flag,expected", [(None, 0), (False, 0), (True, 2)])
def test_checkpoint_upload_is_opt_in(data_dir, tmp_path, monkeypatch, flag, expected):
    from spectral_lejepa.utils import wandb as wb
    calls = []
    monkeypatch.setattr(wb, "init_run", lambda *a, **k: FakeRun())
    monkeypatch.setattr(wb, "log_figure", lambda *a, **k: None)
    monkeypatch.setattr(wb, "log_checkpoint", lambda *a, **k: calls.append(a))
    cfg = tiny_cfg(data_dir, tmp_path / "out", name=f"ckpt_{flag}")
    cfg["training"].update(max_steps=4, ckpt_every=2, val_every=4, diag_every=4, log_every=2)
    if flag is None:
        cfg["wandb"].pop("log_checkpoints", None)       # an old config without the key
    else:
        cfg["wandb"]["log_checkpoints"] = flag
    with pytest.warns(UserWarning):
        train(cfg)
    assert len(calls) == expected
```

- [ ] **Step 4: Run it to verify it fails.**
Run: `uv run pytest tests/test_train_smoke.py -k checkpoint_upload -v`
Expected: FAIL. With the flag unset or false, uploads still happen (2 calls).

- [ ] **Step 5: Implement.**
  - In `trainer.py`'s `checkpoint()` helper, change `if run is not None:` (the one before `wb.log_checkpoint`) to `if run is not None and cfg["wandb"].get("log_checkpoints", False):`.
  - In `configs/pretrain.yaml`, under `wandb:`, add `log_checkpoints: false   # checkpoints stay local; W&B gets metrics and figures only`.

- [ ] **Step 6: Run the tests.**
Run: `uv run pytest tests/test_train_smoke.py tests/test_screen.py -v`, then `uv run pytest`.
Expected: all PASS.

- [ ] **Step 7: Commit.** `git add src/spectral_lejepa/training/trainer.py configs/pretrain.yaml scripts/screen.py tests/test_train_smoke.py tests/test_screen.py`, then `git commit -m "Stop uploading checkpoints to W&B by default; harden the screen launcher"` plus the trailer.

---

### Task 4: Re-score existing checkpoints (controller; stop and report)

- [ ] **Step 1:** Check `git status --short` is empty. Pick a free CUDA device (nvidia-smi index k ≥ 4 is CUDA index k−1).
- [ ] **Step 2: Score the baseline on all 9 sets, with the random control.**
  `bash -ic 'export PATH="$HOME/.local/bin:$PATH"; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<k>; nohup uv run python -m scripts.evaluate --checkpoint outputs/lejepa_baseline_20261002-161540/checkpoint_last.pt evaluation.n_jobs=16 > outputs/eval_fm_baseline.log 2>&1 &'`
  Wait on the `[evaluate] scorecard:` line, not on pgrep.
- [ ] **Step 3: Re-score the 9 screen arms on labeled_data with the three readouts.** For each `outputs/screen-1/<arm>_*/checkpoint_last.pt`, run `python -m scripts.evaluate --checkpoint <ckpt> --config configs/eval_screen.yaml experiment.output_dir=outputs/screen-1/eval_readouts`. Run up to 3 at once, sharing CPUs. Then run `python -m scripts.screen --summarize_only` with a copy of `configs/screen.yaml` whose `eval_config` points at an eval config with `experiment.output_dir: outputs/screen-1/eval_readouts`.
- [ ] **Step 4: Stop and report.** Report:
  - the scorecard for the model and the random control, per set, with the best readout and block;
  - the screen table under the new readouts;
  - a short recommendation feeding sub-project B (architecture, readout and depth to target).
