import pytest
import torch

from spectral_lejepa.models.tokenizer import PatchTokenizer, patch_bounds


def test_patch_widths():
    assert [hi - lo for lo, hi in patch_bounds()] == [11] * 5 + [10] * 19


def test_every_sample_in_exactly_one_patch():
    covered = [i for lo, hi in patch_bounds() for i in range(lo, hi)]
    assert covered == list(range(245))


def test_patchify_values_and_padding():
    tok = PatchTokenizer(dim=8)
    x = torch.arange(245, dtype=torch.float32)[None]
    p = tok.patchify(x)
    assert p.shape == (1, 24, 11)
    assert torch.equal(p[0, 0], torch.arange(0, 11).float())
    assert torch.equal(p[0, 5, :10], torch.arange(55, 65).float()) and p[0, 5, 10] == 0
    assert torch.equal(p[0, -1, :10], torch.arange(235, 245).float()) and p[0, -1, 10] == 0


def test_output_shape():
    assert PatchTokenizer(dim=32)(torch.randn(3, 245)).shape == (3, 24, 32)


def test_wrong_length_raises():
    with pytest.raises(ValueError, match="245"):
        PatchTokenizer(dim=8)(torch.randn(2, 240))
