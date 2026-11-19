#!/usr/bin/env python
"""LeJEPA on measured spectra, standalone: pretrain the measured-only encoder (the report's bm100 recipe,
Appendix J) and score its block-2 features with the same nested-CV parameter_0 probe as raw_baseline.py.
Needs torch (2.8 used), numpy, scipy, scikit-learn (1.6.1), soundfile and joblib; one GPU; no other repo file.

    python lejepa_baseline.py --data-root /path/to/nova_data [--fraction 1.0 | 0.25 | 0.0625] [--seed 107]
                              [--components 0 | 0,1 | 0,1,2] [--sets ...] [--jobs 8] [--output r.json]
                              [--out-dir runs] [--checkpoint C.pt] [--steps N] [--resume runs/<run>/checkpoint_last.pt]

Data layout as raw_baseline.py: <data-root>/labeled_regression/dataset<D>/ (train.tsv + valid.tsv fairseq
manifests, labels.tsv, wavs under wav/, wavs/ or the set directory) and <data-root>/labeled_data/.

Pretraining (labels are never read):
- spectra: every wav listed in labeled_regression/dataset*/{train,valid}.tsv (deduplicated, train first, sets in
  sorted order) except the 122 rows our packed array holds out for validation (HELD_OUT), 147,662 rows in all;
  each normalized with the global mean / std of our packed pretraining array (GLOBAL_MEAN, GLOBAL_STD);
- `--fraction F`: a seeded share F of each set's sample groups (all components of one spectrum index), selection
  seed 0, smaller fractions subsets of larger ones -- the measured-budget arms bm100 / bm25 / bm6 (F = 1, .25, .0625);
- model and recipe: patch ViT (48 patches, dim 256, depth 6) with a 4-block predictor, random 75% masking, masked
  MSE + SIGReg (lambda .05) + 2 pooled global views, AdamW lr 5e-4, wd .05, batch 256, 248,995 steps with 49,799
  warmup steps and cosine decay to 1e-3 x lr, fp16 autocast; `--seed` 107 and 108 are the report's two members.
  The random streams (init, shuffle, masks, SIGReg slices, 2,048-row validation every 2,735 steps) follow our
  trainer call for call, so a run is reproducible; GPU kernels may still differ in the last bits between machines.
- checkpoints every 10,000 steps in <out-dir>/lejepa_f<F>_s<seed>/ (checkpoint_last.pt resumes with --resume;
  like our trainer, a resumed run restarts the epoch's shuffle stream, so it is not bitwise an uninterrupted run).

Evaluation: block-2 flat features (48 tokens x 256 = 12,288 per spectrum after the second transformer block),
nested CV exactly as raw_baseline.py (2x5 outer, 5 inner, 12 recipes, seed 42), beside the raw spectra and a
random-weight encoder of the same architecture (torch seed 0). `--components 0,1[,2]` scores each component and the
joined vector [f_c0 | f_c1 | ...] (Appendix K). `--eval-fraction` keeps a seeded share of each set's labeled samples
as raw_baseline.py --fraction does. `--checkpoint` skips pretraining and evaluates a saved encoder (ours or this
script's).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import time
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=LinAlgWarning)

LENGTH = 245
GLOBAL_MEAN, GLOBAL_STD = 0.3784184896260804, 0.5125814032129958   # our packed pretraining array (stats.json)
MODEL = dict(num_patches=48, dim=256, depth=6, heads=8, mlp_dim=1024, dropout=0.0, predictor_dim=192,
             predictor_depth=4, predictor_heads=6, predictor_mlp_dim=768)
TRAIN = dict(batch_size=256, max_steps=248995, warmup_steps=49799, learning_rate=5e-4, weight_decay=0.05,
             final_lr_ratio=1e-3, mask_ratio=0.75, lambda_sigreg=0.05, sigreg_num_slices=1024, sigreg_knots=17,
             sigreg_t_max=3.0, global_weight=1.0, global_views=2, global_keep=0.5, val_every=2735, valid_rows=2048,
             ckpt_every=10000, log_every=500, num_workers=4)
FRACTION_SEED = 0        # sample-group selection seed for --fraction (pretraining)
BLOCK = 2                # readout: tokens after the second transformer block, flattened
# rows of labeled_regression our packed array holds out for validation (dataset:component:spec); never trained on
HELD_OUT = set("""
0055:25:479 0055:5:128 0055:2:115 0055:22:454 0055:28:307 0055:25:103 0055:9:463 0055:15:41 0055:14:18
0055:31:114 0055:17:206 0055:28:454 0055:25:118 0106:2:571 0106:10:813 0106:0:499 0106:7:129 0106:5:1655
0106:11:648 0106:25:147 0106:6:486 0106:9:2706 0106:8:1020 0106:9:1876 0106:10:1247 0106:7:319 0106:8:42
0106:26:661 0106:8:1617 0106:2:2059 0106:30:186 0106:0:2170 0106:9:2940 0106:2:3233 0106:8:2031 0106:20:412
0106:8:1526 0106:5:2253 0106:14:231 0106:26:658 0106:9:1086 0106:9:2154 0106:3:2778 0106:2:3350 0106:8:755
0106:8:1104 0106:20:238 0106:5:2327 0107:21:17 0108:1:1999 0108:28:1013 0108:26:1392 0108:28:90 0108:1:1444
0108:29:2220 0108:28:473 0108:1:228 0108:29:1045 0108:0:1867 0108:27:1748 0108:27:1475 0108:30:1824 0108:26:1302
0109:31:189 0109:1:719 0111:28:17 0112:4:69 0112:24:25 0112:8:112 0112:1:148 0112:16:347 0112:17:499 0112:4:480
0112:3:376 0112:7:387 0112:4:365 0112:15:396 0112:7:103 0112:24:42 0112:24:29 0112:12:333 0112:23:330
0112:27:209 0113:0:927 0113:1:1499 0113:28:2 0113:0:1428 0114:0:552 0114:1:327 0115:31:90 0115:26:240 0115:1:201
0115:1:218 0117:29:29 0117:19:124 0118:27:174 0119:7:189 0119:11:123 0119:3:104 0120:19:160 0120:26:357
0120:28:320 0120:19:357 0120:1:514 0120:1:557 0120:4:46 0120:30:428 0120:19:569 0120:5:54 0120:31:281
0120:19:854 0120:1:684 0120:16:90 0120:30:142 0120:13:127 0120:9:23 0120:29:941 0120:14:963 0120:21:57
0120:23:214 0120:29:819 0120:5:416""".split())

# evaluation protocol (as raw_baseline.py)
SEED = 42
MAX_SAMPLES = 5000
MIN_N = 20
N_REPEATS, N_FOLDS, N_INNER = 2, 5, 5
NORMALIZERS = ("none", "standardize", "whiten", "whiten8", "whiten32", "whiten128")
PROBES = ("ridgecv", "ols")
RECIPES = tuple((n, p) for n in NORMALIZERS for p in PROBES)
WHITEN_SAMPLES_PER_DIRECTION = 2
SETS = ("labeled_data", "dataset0055", "dataset0106", "dataset0109", "dataset0112", "dataset0113", "dataset0114",
        "dataset0120")
PATTERN = re.compile(r"dataset(\d+)_comp(\d+)_spec_(\d+)\.wav")


# ---------------------------------------------------------------- model

def patch_bounds(sequence_length, num_patches):
    parts = np.array_split(np.arange(sequence_length), num_patches)
    return [(int(p[0]), int(p[-1]) + 1) for p in parts]


class PatchTokenizer(nn.Module):
    """[B, 245] -> [B, T, dim]: contiguous patches (zero padded to the widest), linear projection, learned positions."""

    def __init__(self, sequence_length, num_patches, dim):
        super().__init__()
        self.sequence_length = sequence_length
        self.bounds = patch_bounds(sequence_length, num_patches)
        self.patch_width = max(hi - lo for lo, hi in self.bounds)
        # index[p, j] = sample index of position j in patch p; positions past a patch's end point at the appended zero
        index = torch.full((num_patches, self.patch_width), sequence_length, dtype=torch.long)
        for p, (lo, hi) in enumerate(self.bounds):
            index[p, : hi - lo] = torch.arange(lo, hi)
        self.register_buffer("index", index, persistent=False)
        self.proj = nn.Linear(self.patch_width, dim)
        self.pos_embed = nn.Parameter(torch.zeros(num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return self.proj(F.pad(x, (0, 1))[:, self.index]) + self.pos_embed


def gather_tokens(x, idx):
    """x [B, N, D], idx [B, K] -> [B, K, D]."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def transformer_blocks(dim, depth, heads, mlp_dim, dropout):
    return nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout, activation="gelu",
                                                     batch_first=True, norm_first=True) for _ in range(depth)])


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.blocks = transformer_blocks(dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def prefix(self, tokens, k):
        """State after the first k blocks (no final norm): the block-k readout."""
        for block in self.blocks[:k]:
            tokens = block(tokens)
        return tokens


class Predictor(nn.Module):
    """Encoder embeddings at masked positions from the context embeddings plus one mask token per position."""

    def __init__(self, num_patches, in_dim, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.pos_embed = nn.Parameter(torch.zeros(num_patches, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = transformer_blocks(dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, in_dim)

    def forward(self, context, visible_idx, masked_idx):
        B, n_mask = masked_idx.shape
        ctx = self.embed(context) + self.pos_embed[visible_idx]                      # [B, n_vis, dim]
        queries = self.mask_token.expand(B, n_mask, -1) + self.pos_embed[masked_idx]  # [B, n_mask, dim]
        x = torch.cat([ctx, queries], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, -n_mask:]))                                   # [B, n_mask, in_dim]


class LeJEPA(nn.Module):
    def __init__(self, sequence_length, num_patches, dim, depth, heads, mlp_dim, dropout,
                 predictor_dim, predictor_depth, predictor_heads, predictor_mlp_dim):
        super().__init__()
        self.tokenizer = PatchTokenizer(sequence_length, num_patches, dim)
        self.encoder = Encoder(dim, depth, heads, mlp_dim, dropout)
        self.predictor = Predictor(num_patches, dim, predictor_dim, predictor_depth, predictor_heads,
                                   predictor_mlp_dim, dropout)

    def forward(self, x, masked_idx, visible_idx, view_idxs):
        tokens = self.tokenizer(x)                                            # [B, T, D]
        context = self.encoder(gather_tokens(tokens, visible_idx))            # [B, n_vis, D]
        predicted = self.predictor(context, visible_idx, masked_idx)          # [B, n_mask, D]
        target = self.encoder(tokens)                                         # [B, T, D]
        views = torch.stack([target.mean(1)] + [self.encoder(gather_tokens(tokens, idx)).mean(1)
                                                for idx in view_idxs])        # [V+1, B, D]
        return predicted, gather_tokens(target, masked_idx), target, views


def build_model():
    return LeJEPA(sequence_length=LENGTH, **MODEL)


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer (LeJEPA; lightly 1.5.26's SIGReg, single process): Epps-Pulley
    statistic of random 1-D projections against N(0, 1), trapezoidal rule over [0, t_max]."""

    def __init__(self, knots, t_max, num_vectors):
        super().__init__()
        self.num_vectors = num_vectors
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj [..., N, C] -> scalar."""
        num_samples = proj.size(-2)
        A = torch.randn(proj.size(-1), self.num_vectors, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        cos_mean = x_t.cos().sum(-3) / num_samples
        sin_mean = x_t.sin().sum(-3) / num_samples
        err = (cos_mean - self.phi.to(dtype=x_t.dtype)).square() + sin_mean.square()
        return ((err @ self.weights.to(dtype=err.dtype)) * num_samples).mean()


class Objective(nn.Module):
    """Masked MSE + lambda SIGReg over tokens + global_weight x ((1 - lambda) view invariance + lambda SIGReg over
    pooled views), all in fp32."""

    def __init__(self):
        super().__init__()
        self.lam, self.global_weight = TRAIN["lambda_sigreg"], TRAIN["global_weight"]
        self.sigreg = SIGReg(TRAIN["sigreg_knots"], TRAIN["sigreg_t_max"], TRAIN["sigreg_num_slices"])

    def forward(self, predicted, target_masked, target, views):
        with torch.autocast(device_type=target.device.type, enabled=False):
            mse = F.mse_loss(predicted.float(), target_masked.float())
            reg = self.sigreg(target.float().transpose(0, 1))   # [T, B, D]
            views = views.float()
            inv = ((views - views.mean(0, keepdim=True)) ** 2).mean()
            greg = self.sigreg(views)                            # [V+1, B, D]
            loss = mse + self.lam * reg + self.global_weight * ((1 - self.lam) * inv + self.lam * greg)
        return {"loss": loss, "mse": mse, "sigreg": reg, "global_inv": inv, "global_sigreg": greg}


def random_mask(batch_size, num_patches, mask_ratio, generator, device):
    """(masked_idx, visible_idx), sorted, round(mask_ratio x num_patches) masked per sample (CPU generator)."""
    n_mask = round(mask_ratio * num_patches)
    order = torch.rand(batch_size, num_patches, generator=generator).argsort(dim=1)
    return order[:, :n_mask].sort(dim=1).values.to(device), order[:, n_mask:].sort(dim=1).values.to(device)


def masks_and_views(batch_size, generator, device):
    masked_idx, visible_idx = random_mask(batch_size, MODEL["num_patches"], TRAIN["mask_ratio"], generator, device)
    views = [random_mask(batch_size, MODEL["num_patches"], 1 - TRAIN["global_keep"], generator, device)[1]
             for _ in range(TRAIN["global_views"])]
    return masked_idx, visible_idx, views


# ---------------------------------------------------------------- pretraining data

def set_dir(root, name):
    return os.path.join(root, "labeled_data" if name == "labeled_data" else os.path.join("labeled_regression", name))


def _wav_root(d):
    for sub in ("wav", "wavs"):
        if glob.glob(os.path.join(d, sub, "*.wav")):
            return os.path.join(d, sub)
    return d


def manifest_names(path):
    """Wav names of a fairseq manifest (line 1 is the root, then `name<TAB>num_samples`)."""
    with open(path) as f:
        f.readline()
        return [line.split("\t")[0] for line in f if line.strip()]


def pretraining_spectra(root, fraction):
    """(X [n, 245] float32 raw, n_groups kept): the measured pretraining rows in our packed order."""
    sets = sorted(d for d in glob.glob(os.path.join(root, "labeled_regression", "dataset*"))
                  if os.path.exists(os.path.join(d, "train.tsv")))
    rng = np.random.default_rng(FRACTION_SEED)
    rows, n_groups = [], 0
    for d in sets:
        names = []
        for split in ("train", "valid"):
            if os.path.exists(os.path.join(d, f"{split}.tsv")):
                names += manifest_names(os.path.join(d, f"{split}.tsv"))
        names = [n for n in dict.fromkeys(names)
                 if "{}:{}:{}".format(*PATTERN.fullmatch(os.path.basename(n)).groups()) not in HELD_OUT]
        wroot = _wav_root(d)
        X = np.empty((len(names), LENGTH), np.float32)
        for i, n in enumerate(names):
            x, _ = sf.read(os.path.join(wroot, os.path.basename(n)), dtype="float32")
            if x.shape != (LENGTH,):
                raise ValueError(f"{n}: expected {LENGTH} samples, got shape {x.shape}")
            X[i] = x
        ok = (X.std(axis=1) >= 1e-6) & np.isfinite(X).all(axis=1)   # constant / non-finite rows are never trained on
        X, names = X[ok], [n for n, k in zip(names, ok) if k]
        if not len(names):
            continue
        keys = np.array(["dataset{}/{}".format(*PATTERN.fullmatch(os.path.basename(n)).group(1, 3)) for n in names])
        groups = rng.permutation(np.unique(keys))
        chosen = groups[:max(1, math.ceil(fraction * len(groups)))]
        rows.append(X[np.isin(keys, chosen)])
        n_groups += len(chosen)
        print(f"  {os.path.basename(d)}: {len(names)} spectra, {len(chosen)}/{len(groups)} sample groups kept, "
              f"{len(rows[-1])} rows", flush=True)
    return np.concatenate(rows), n_groups


class Rows(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i].copy()) - GLOBAL_MEAN) / GLOBAL_STD


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model, weight_decay):
    """No weight decay on biases, norms, positional embeddings and the mask token."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.endswith("pos_embed") or name.endswith("mask_token") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step, warmup, total, final_ratio):
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validation_loss(model, objective, valid, seed, device):
    """Mean loss over the fixed validation rows (mask seed fixed). Also keeps the SIGReg random stream in step
    with our trainer, which validates on 2,048 rows of its packed array at the same steps."""
    model.eval()
    gen, bs, total, n = torch.Generator().manual_seed(seed), TRAIN["batch_size"], 0.0, 0
    for i in range(0, len(valid) - bs + 1, bs):
        x = valid[i:i + bs].to(device)
        masked_idx, visible_idx, views = masks_and_views(len(x), gen, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x, masked_idx, visible_idx, views)
        total += objective(*out)["loss"].item()
        n += 1
    model.train()
    return total / n


def pretrain(a, device):
    print(f"pretraining spectra (fraction {a.fraction}):", flush=True)
    X, n_groups = pretraining_spectra(a.data_root, a.fraction)
    steps = a.steps or TRAIN["max_steps"]
    run_dir = os.path.join(a.out_dir, f"lejepa_f{a.fraction:g}_s{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    config = {"model": dict(MODEL), "data": {"sequence_length": LENGTH, "normalization": "global"},
              "training": {**TRAIN, "max_steps": steps}, "experiment": {"seed": a.seed},
              "derived": {"global_stats": {"mean": GLOBAL_MEAN, "std": GLOBAL_STD}, "measured_fraction": a.fraction,
                          "n_measured_groups": n_groups, "train_samples": len(X)}}
    print(f"{len(X)} rows, {n_groups} sample groups; {steps} steps -> {run_dir}", flush=True)

    seed, bs = a.seed, TRAIN["batch_size"]
    set_seed(seed)
    loader = DataLoader(Rows(X), batch_size=bs, shuffle=True, drop_last=True, num_workers=TRAIN["num_workers"],
                        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
                        persistent_workers=TRAIN["num_workers"] > 0)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError(f"{len(X)} rows is fewer than one batch ({bs})")
    pick = np.random.default_rng(0).choice(len(X), TRAIN["valid_rows"], replace=len(X) < TRAIN["valid_rows"])
    valid = torch.stack([Rows(X)[i] for i in np.sort(pick)])        # monitoring only (trained on too)
    model = build_model().to(device)
    objective = Objective().to(device)
    optimizer = torch.optim.AdamW(param_groups(model, TRAIN["weight_decay"]), lr=TRAIN["learning_rate"])
    warmup = TRAIN["warmup_steps"] if TRAIN["warmup_steps"] < steps else max(1, steps // 10)
    scheduler = LambdaLR(optimizer, lambda s: lr_factor(s, warmup, steps, TRAIN["final_lr_ratio"]))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    mask_gen = torch.Generator().manual_seed(seed + 1)
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        scheduler.load_state_dict(ck["scheduler"])
        rng = ck["rng_state"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        mask_gen.set_state(rng["mask"])
        if amp and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])
        step = ck["step"]
        print(f"resuming at step {step}", flush=True)

    def save(path):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "epoch": step / steps_per_epoch,
                    "config": config, "rng_state": {
                        "torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                        "mask": mask_gen.get_state(), "cuda": torch.cuda.get_rng_state_all() if amp else None}},
                   path + ".tmp")
        os.replace(path + ".tmp", path)

    model.train()
    t0, last = time.time(), False
    while not last:
        skip = step % steps_per_epoch
        for batch_index, x in enumerate(loader):
            if batch_index < skip:
                continue
            x = x.to(device, non_blocking=True)
            masked_idx, visible_idx, views = masks_and_views(len(x), mask_gen, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = model(x, masked_idx, visible_idx, views)
            losses = objective(*out)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            last = step == (a.stop_step or steps)
            if step % TRAIN["log_every"] == 0 or last:
                loss = losses["loss"].item()
                if not math.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                print(f"step {step}/{steps} loss {loss:.5f} mse {losses['mse'].item():.5f} "
                      f"sigreg {losses['sigreg'].item():.4f} lr {scheduler.get_last_lr()[0]:.3g} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if step % TRAIN["val_every"] == 0 or last:
                print(f"step {step} valid/loss {validation_loss(model, objective, valid, seed + 2, device):.5f}",
                      flush=True)
            if step % TRAIN["ckpt_every"] == 0 or last:
                save(os.path.join(run_dir, f"checkpoint_step{step}.pt"))
                save(os.path.join(run_dir, "checkpoint_last.pt"))
            if last:
                break
    return os.path.join(run_dir, "checkpoint_last.pt")


# ---------------------------------------------------------------- evaluation data and features

def load(d, components):
    """(X {component: [n, 245] float32}, y [n], n_all): samples (dataset, spec) that have every listed component
    in labels.tsv and on disk, sorted; n_all = number of such samples before any subsample."""
    files, labels = {}, {}
    root = _wav_root(d)
    with open(os.path.join(d, "labels.tsv")) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2 or (m := PATTERN.match(parts[0])) is None:
                continue
            key = (int(m[1]), int(m[3]), int(m[2]))
            if key[2] not in components:
                continue
            path = os.path.join(root, parts[0])
            files[key] = path if os.path.isfile(path) else os.path.join(d, parts[0])
            labels[key] = float(parts[1])
    keys = [k for k in sorted({k[:2] for k in files})
            if all((*k, c) in files and os.path.isfile(files[(*k, c)]) for c in components)]
    n_all = len(keys)
    if len(keys) > MAX_SAMPLES:
        idx = np.sort(np.random.default_rng(SEED).choice(len(keys), MAX_SAMPLES, replace=False))
        keys = [keys[i] for i in idx]
    for k in keys:
        if len({labels[(*k, c)] for c in components}) != 1:
            raise ValueError(f"dataset {k[0]}, spec {k[1]}: components disagree on the label")
    X = {}
    for c in components:
        rows = []
        for k in keys:
            data = np.asarray(sf.read(files[(*k, c)], dtype="float32")[0]).flatten()
            row = np.zeros(LENGTH, np.float32)
            row[:min(len(data), LENGTH)] = data[:LENGTH]
            rows.append(row)
        X[c] = np.stack(rows)
    y = np.array([labels[(*k, components[0])] for k in keys], np.float64)
    return X, y, n_all


def take_fraction(n, fraction):
    """Sorted indices of the kept samples: the first ceil(fraction x n) of one seeded permutation."""
    if fraction >= 1:
        return np.arange(n)
    perm = np.random.default_rng(FRACTION_SEED).permutation(n)
    return np.sort(perm[:max(1, math.ceil(fraction * n))])


def load_encoder(path):
    """(model, (mean, std)) from a checkpoint of this script or of our trainer (global normalization only)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg["data"].get("normalization") != "global":
        raise ValueError(f"{path}: trained with {cfg['data'].get('normalization')!r} normalization, expected global")
    if any(cfg["model"][k] != v for k, v in MODEL.items()):
        raise ValueError(f"{path}: model config differs from this script's")
    model = build_model()
    model.load_state_dict(ck["model"])
    g = cfg["derived"]["global_stats"]
    return model, (g["mean"], g["std"])


@torch.no_grad()
def block_features(model, X, stats, device, batch_size=64):
    """Block-2 flat features [n, 48 x 256] of raw spectra X [n, 245], fp32."""
    model.eval().to(device)
    Z = torch.from_numpy(((X - stats[0]) / stats[1]).astype(np.float32))
    out = []
    for i in range(0, len(Z), batch_size):
        h = model.encoder.prefix(model.tokenizer(Z[i:i + batch_size].to(device)), BLOCK)
        out.append(h.reshape(len(h), -1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- probes (as raw_baseline.py)

def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _standardize(X_tr, X_te):
    X_tr32 = np.asarray(X_tr).astype(np.float32, copy=False)
    mu = X_tr32.mean(axis=0).astype(np.float32)
    sd = X_tr32.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return [((np.asarray(X, dtype=np.float32) - mu) / sd).astype(np.float32) for X in (X_tr, X_te)]


def _whiten(X_tr, X_te, seed):
    """Exact PCA whitening fit on X_tr, at most one direction per 2 training rows, dead directions dropped."""
    X64 = np.asarray(X_tr, dtype=np.float64)
    n, d = X64.shape
    k = min(d, max(1, n // WHITEN_SAMPLES_PER_DIRECTION))
    pca = PCA(n_components=k, whiten=True, svd_solver="full", random_state=seed).fit(X64)
    ev = pca.explained_variance_
    tol = (np.finfo(np.float64).eps * max(n, d)) ** 2
    k_keep = max(1, int(np.sum(ev > tol * ev[0])) if ev.size else 0)
    return [pca.transform(np.asarray(X, dtype=np.float64))[:, :k_keep].astype(np.float32) for X in (X_tr, X_te)]


def fold_transforms(X_tr, X_te, seed):
    out = {"none": (np.asarray(X_tr, np.float32), np.asarray(X_te, np.float32)),
           "standardize": tuple(_standardize(X_tr, X_te))}
    Ztr, Zte = _whiten(X_tr, X_te, seed)
    for name in NORMALIZERS[2:]:
        k = Ztr.shape[1] if name == "whiten" else min(Ztr.shape[1], int(name[len("whiten"):]))
        out[name] = (Ztr[:, :k], Zte[:, :k])
    return out


def recipe_predictions(X_tr, y_tr, X_te, recipes, seed):
    T = fold_transforms(X_tr, X_te, seed)
    out = {}
    for norm, probe in recipes:
        m = RidgeCV(alphas=np.logspace(-3, 3, 20)) if probe == "ridgecv" else LinearRegression()
        m.fit(T[norm][0], y_tr)
        out[(norm, probe)] = np.asarray(m.predict(T[norm][1])).ravel()
    return out


def inner_folds(n, seed, grouped):
    """Row-level KFold (comp0 protocol) or the component study's seeded group folds (one group per sample)."""
    if not grouped:
        yield from KFold(N_INNER, shuffle=True, random_state=seed).split(np.zeros(n))
        return
    for chunk in np.array_split(np.random.default_rng(seed).permutation(n), N_INNER):
        test = np.isin(np.arange(n), chunk)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def outer_fold(X, y, tr, te, seed, grouped):
    """(held-out prediction, chosen recipe, inner R²) for one arm: recipe picked by inner CV on tr, refit on tr."""
    with threadpool_limits(limits=1):
        Xtr, ytr = X[tr], y[tr]
        preds = {rc: np.zeros(len(tr)) for rc in RECIPES}
        for itr, ite in inner_folds(len(tr), seed, grouped):
            P = recipe_predictions(Xtr[itr], ytr[itr], Xtr[ite], RECIPES, seed)
            for rc in RECIPES:
                preds[rc][ite] = P[rc]
        inner = {rc: r2(ytr, p) for rc, p in preds.items()}
        best = max(RECIPES, key=lambda rc: inner[rc])
        return recipe_predictions(Xtr, ytr, X[te], [best], seed)[best], best, inner[best]


def nested_cv(arms, y, grouped, jobs):
    n = len(y)
    folds = [(r, f, tr, te) for r in range(N_REPEATS)
             for f, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(np.zeros(n)))]
    tasks = [(a, r, f, tr, te) for a in arms for r, f, tr, te in folds]
    outs = Parallel(n_jobs=jobs)(delayed(outer_fold)(arms[a], y, tr, te, SEED + 1000 + 10 * r + f, grouped)
                                 for a, r, f, tr, te in tasks)
    oof = {a: np.zeros((N_REPEATS, n)) for a in arms}
    chosen = {a: [] for a in arms}
    for (a, r, f, tr, te), (pred, (norm, probe), ir2) in zip(tasks, outs):
        oof[a][r, te] = pred
        chosen[a].append({"repeat": r, "fold": f, "norm": norm, "probe": probe, "inner_r2": ir2})
    return oof, chosen


def bootstrap_sd(y, P, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.std([np.mean([r2(y[i], p[i]) for p in P])
                         for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]))


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", required=True, help="directory holding labeled_regression/ and labeled_data/")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of each set's sample groups to pretrain on")
    ap.add_argument("--seed", type=int, default=107, help="pretraining seed (the report's members: 107, 108)")
    ap.add_argument("--components", default="0", help="0 (single component) or e.g. 0,1 / 0,1,2 (joined)")
    ap.add_argument("--eval-fraction", type=float, default=1.0, help="share of each set's labeled samples to score")
    ap.add_argument("--sets", nargs="+", help="default: all eight sets (components 0) or the seven "
                                              "labeled_regression sets (joined components)")
    ap.add_argument("--out-dir", default="lejepa_runs", help="pretraining checkpoints go to <out-dir>/lejepa_f<F>_s<seed>")
    ap.add_argument("--checkpoint", help="skip pretraining; evaluate this encoder checkpoint")
    ap.add_argument("--resume", help="continue pretraining from this checkpoint_last.pt")
    ap.add_argument("--steps", type=int, help=f"pretraining steps (default {TRAIN['max_steps']}; fewer only for tests)")
    ap.add_argument("--stop-step", type=int, help="stop at this step, LR schedule kept at full length (tests)")
    ap.add_argument("--no-controls", action="store_true", help="skip the raw and random-encoder arms")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="outer folds run in parallel")
    ap.add_argument("--output", help="write per-set results (scores, chosen recipes, OOF predictions) as JSON")
    a = ap.parse_args(argv)
    comps = tuple(int(c) for c in a.components.split(","))
    for name in ("fraction", "eval_fraction"):
        if not 0 < getattr(a, name) <= 1:
            ap.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    device = torch.device(a.device)
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    ckpt = a.checkpoint or pretrain(a, device)
    model, stats = load_encoder(ckpt)
    torch.manual_seed(0)
    random_model = build_model()

    joined = len(comps) > 1
    sets = a.sets or [s for s in SETS if not (joined and s == "labeled_data")]
    scores = ([f"comp{c}" for c in comps] + ["concat"]) if joined else ["comp0"]
    last = scores[-1]
    shown = scores + ([] if a.no_controls else [f"raw:{last}", f"random:{last}"])
    print(f"LeJEPA block-{BLOCK} flat, {ckpt}: components {comps}, eval fraction {a.eval_fraction}, seed {SEED}, "
          f"{N_REPEATS}x{N_FOLDS} outer / {N_INNER} inner folds, {len(RECIPES)} recipes")
    print(f"{'set':<13}{'n_all':>7}{'n':>6}  " + "  ".join(f"{s + ' R²':>18}{'RMSE':>9}" for s in shown))
    results = {}
    for name in sets:
        X, y, n_all = load(set_dir(a.data_root, name), comps)
        keep = take_fraction(len(y), a.eval_fraction)
        X, y = {c: v[keep] for c, v in X.items()}, y[keep]
        if len(y) < MIN_N:
            print(f"{name:<13}{n_all:>7}{len(y):>6}  skipped: fewer than {MIN_N} samples")
            continue
        feats = {c: block_features(model, X[c], stats, device) for c in comps}
        arms = {f"comp{c}": feats[c] for c in comps}
        if joined:
            arms["concat"] = np.concatenate([feats[c] for c in comps], axis=1)
        if not a.no_controls:
            arms[f"raw:{last}"] = np.concatenate([X[c] for c in comps], axis=1)
            arms[f"random:{last}"] = np.concatenate([block_features(random_model, X[c], stats, device)
                                                     for c in comps], axis=1)
        oof, chosen = nested_cv(arms, y, grouped=joined, jobs=a.jobs)
        res = {s: {"r2": float(np.mean([r2(y, p) for p in oof[s]])), "r2_sd": bootstrap_sd(y, oof[s]),
                   "rmse": float(np.mean([rmse(y, p) for p in oof[s]]))} for s in shown}
        print(f"{name:<13}{n_all:>7}{len(y):>6}  " + "  ".join(
            f"{res[s]['r2']:>11.3f} ±{res[s]['r2_sd']:.3f}{res[s]['rmse']:>9.4g}" for s in shown), flush=True)
        results[name] = {"n_all": n_all, "n": int(len(y)), "kept": keep.tolist(), "scores": res,
                         "chosen": chosen, "oof": {s: oof[s].tolist() for s in shown}, "y": y.tolist()}
    if results:
        print(f"{'mean':<26}" + "  ".join(f"{np.mean([r['scores'][s]['r2'] for r in results.values()]):>11.3f}"
                                          f"{'':>7}{np.mean([r['scores'][s]['rmse'] for r in results.values()]):>9.4g}"
                                          for s in shown))
    if a.output:
        with open(a.output, "w") as f:
            json.dump({"protocol": {"checkpoint": ckpt, "fraction": a.fraction, "seed": a.seed, "components": comps,
                                    "eval_fraction": a.eval_fraction, "eval_seed": SEED, "block": BLOCK,
                                    "recipes": [f"{n}+{p}" for n, p in RECIPES]},
                       "sets": results}, f)


if __name__ == "__main__":
    main()
