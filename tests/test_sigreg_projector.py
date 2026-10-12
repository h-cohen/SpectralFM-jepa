import copy
import torch

from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.models.masking import random_mask
from spectral_lejepa.training.loss import LeJEPAObjective
from spectral_lejepa.training.trainer import load_init_checkpoint, initialize_weights

SMALL=dict(num_patches=8,dim=16,depth=6,heads=4,mlp_dim=32,dropout=0.,predictor_dim=16,
           predictor_depth=1,predictor_heads=4,predictor_mlp_dim=32,projector='none',
           projector_hidden_dim=32,projector_dim=8)


def test_control_and_projected_modes_keep_backbone_outputs_and_rng_identical():
    def model(mode):
        torch.manual_seed(4)
        return build_model({**SMALL,'sigreg_projector':mode,'sigreg_projector_hidden_dim':32,'sigreg_projector_dim':8},245).eval()
    c,p=model('control'),model('mlp')
    assert torch.equal(c.sigreg_projector[0].weight,p.sigreg_projector[0].weight)
    x=torch.randn(8,245);masked,visible=random_mask(8,8,.75,torch.Generator().manual_seed(7))
    views=[visible]
    a,b=c(x,masked,visible,views),p(x,masked,visible,views)
    for k in ('predicted','target','target_masked','views'):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    assert 'sigreg_target' not in a
    assert b['sigreg_target'].shape==(8,8,8) and b['sigreg_views'].shape==(2,8,8)
    assert b['sigreg_target'].dtype==torch.float32
    assert not any(isinstance(m,(torch.nn.BatchNorm1d,torch.nn.Dropout)) for m in p.sigreg_projector.modules())


def test_only_sigreg_uses_projected_inputs_and_gradients_reach_encoder():
    p=build_model({**SMALL,'sigreg_projector':'mlp','sigreg_projector_hidden_dim':32,'sigreg_projector_dim':8},245)
    x=torch.randn(8,245);masked,visible=random_mask(8,8,.75,torch.Generator().manual_seed(1))
    o=p(x,masked,visible,[visible]);loss=LeJEPAObjective(.05,num_slices=8,global_weight=1.)
    a=loss(o['predicted'],o['target_masked'],o['target'],o['views'])
    b=loss(o['predicted'],o['target_masked'],o['target'],o['views'],reg_target=o['sigreg_target'],reg_views=o['sigreg_views'])
    torch.testing.assert_close(a['mse_loss'],b['mse_loss'],rtol=0,atol=0)
    torch.testing.assert_close(a['global_inv_loss'],b['global_inv_loss'],rtol=0,atol=0)
    (b['sigreg_loss']+b['global_sigreg_loss']).backward()
    assert p.sigreg_projector[0].weight.grad.abs().sum()>0
    assert p.encoder.blocks[1].linear1.weight.grad.abs().sum()>0


def test_warm_start_initializes_only_new_head_and_keeps_base_exact(tmp_path):
    old=build_model(SMALL,245)
    cfg={'model':SMALL,'data':{'normalization':'global'}}
    path=tmp_path/'old.pt';torch.save({'model':old.state_dict(),'config':cfg,'step':10},path)
    newcfg=copy.deepcopy(cfg);newcfg['model'].update(sigreg_projector='mlp',sigreg_projector_hidden_dim=32,sigreg_projector_dim=8)
    initial=load_init_checkpoint(path,newcfg);new=build_model(newcfg['model'],245)
    initialize_weights(new,initial,resume=False)
    for k,v in old.state_dict().items():torch.testing.assert_close(new.state_dict()[k],v,rtol=0,atol=0)
    broken=copy.deepcopy(initial);broken['ckpt']['model'].pop('encoder.norm.weight')
    import pytest
    with pytest.raises(ValueError,match='initialization'):initialize_weights(new,broken,resume=False)
