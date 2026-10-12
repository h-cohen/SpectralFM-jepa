"""Per-term encoder gradient measurements with preserved weights, gradients and RNG."""
import hashlib
import torch
from ..models.masking import make_mask, random_mask

SEED = 4000

def _hash_tensors(values):
    h = hashlib.sha256()
    for value in values:
        if value is None: h.update(b"none")
        else:
            a = value.detach().cpu().contiguous().numpy()
            h.update(a.dtype.str.encode() + repr(a.shape).encode() + a.tobytes())
    return h.hexdigest()
def _vector(grads, params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p, g in zip(params, grads)])
def audit_batch(model, objective, x, masking_cfg, global_cfg, *, seed=SEED, device="cpu"):
    if model.training:
        raise ValueError("model must be in eval mode for the frozen gradient audit")
    if objective.regularizer != "sigreg":
        raise ValueError("gradient audit expects the final SIGReg objective")
    device = torch.device(device)
    x = x.to(device)
    params = {"block2": list(model.encoder.blocks[1].parameters()), "block6": list(model.encoder.blocks[5].parameters())}
    flat_params = params["block2"] + params["block6"]
    weight_before = _hash_tensors(list(model.parameters()))
    grad_before = _hash_tensors([p.grad for p in model.parameters()])
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda": torch.cuda.default_generators[devices[0]].manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        num_patches = len(model.tokenizer.bounds)
        masked, visible = make_mask(masking_cfg, len(x), num_patches, generator, device)
        views = [random_mask(len(x), num_patches, 1 - global_cfg["keep"], generator, device)[1]
                 for _ in range(global_cfg["views"])]
        output = model(x, masked, visible, views)
        losses = objective(output["predicted"], output["target_masked"], output["target"], output["views"],
                           reg_target=output.get('sigreg_target'),reg_views=output.get('sigreg_views'))
        lam, weight = objective.lambda_sigreg, objective.global_weight
        terms = {
            "mse": losses["mse_loss"],
            "token_sigreg": lam * losses["sigreg_loss"],
            "global_invariance": weight * (1 - lam) * losses["global_inv_loss"],
            "global_sigreg": weight * lam * losses["global_sigreg_loss"],
        }
        vectors = {}
        for name, term in terms.items():
            grads = torch.autograd.grad(term, flat_params, retain_graph=True, allow_unused=True)
            vectors[name] = {g: _vector(grads[o:o + len(ps)], ps) for g, ps, o in (
                ("block2", params["block2"], 0), ("block6", params["block6"], len(params["block2"]))) }
        block_metrics = {}
        for group in params:
            grads = {name: vectors[name][group] for name in terms}
            prediction = grads["mse"]
            other = sum(grads[name] for name in terms if name != "mse")
            denom = prediction.norm() * other.norm()
            block_metrics[group] = {"gradient_norms": {n: float(v.norm().item()) for n, v in grads.items()},
                "gradient_norm_total": float(sum(grads.values()).norm().item()),
                "cosine_prediction_vs_other": float(torch.dot(prediction, other).item() / denom.item()) if denom else 0.0}
        term_values = {name: float(value.detach().item()) for name, value in terms.items()}
        total = float(losses["loss"].detach().item())
        if abs(sum(term_values.values()) - total) > 1e-5 * max(1.0, abs(total)):
            raise AssertionError("weighted four-term sum differs from LeJEPAObjective loss")
    weights_unchanged = weight_before == _hash_tensors(list(model.parameters()))
    grads_unchanged = grad_before == _hash_tensors([p.grad for p in model.parameters()])
    if not weights_unchanged or not grads_unchanged:
        raise AssertionError("autograd audit changed model weights or parameter gradients")
    return {"terms": term_values, "total_loss": total, "blocks": block_metrics,
            "weights_unchanged": weights_unchanged, "parameter_grads_unchanged": grads_unchanged}
