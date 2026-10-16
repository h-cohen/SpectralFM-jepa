import torch

from spectral_lejepa.evaluation.data2vec import is_data2vec, random_like


def test_is_data2vec_tells_fairseq_from_ours(tmp_path):
    torch.save({'cfg': {'model': {'_name': 'data2vec_audio'}}, 'model': {}}, tmp_path / 'd2v.pt')
    torch.save({'config': {}, 'model': {}}, tmp_path / 'ours.pt')
    assert is_data2vec(tmp_path / 'd2v.pt') and not is_data2vec(tmp_path / 'ours.pt')


def test_random_like_reinitialises_and_leaves_the_original():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    with torch.no_grad():
        model[0].weight.fill_(7.0)
    rnd = random_like(model)
    assert torch.all(model[0].weight == 7.0)
    assert not torch.equal(rnd[0].weight, model[0].weight)
    assert torch.equal(random_like(model)[0].weight, rnd[0].weight)   # seeded
