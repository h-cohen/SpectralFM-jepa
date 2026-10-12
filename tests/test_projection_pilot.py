from scripts.projection_pilot import training_config
from spectral_lejepa.training.trainer import schedule_lengths


def test_projector_pilot_pairs_streams_and_schedule_without_changing_backbone_recipe():
    a,b=training_config(301,'control'),training_config(301,'projected')
    assert schedule_lengths(49799,a['training'])==(49799,1000)
    assert a['training']['lambda_sigreg']==b['training']['lambda_sigreg']==.05
    assert a['training']['init_checkpoint'] and a['training']['resume'] is None
    assert a['model']['sigreg_projector']=='control' and b['model']['sigreg_projector']=='mlp'
    for c in (a,b):
        c['experiment'].pop('name');c['model'].pop('sigreg_projector');c['wandb'].pop('tags')
    assert a==b
