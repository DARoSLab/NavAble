optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'query_feat': dict(decay_mult=0.0),
            'query_embed': dict(decay_mult=0.0),
            'level_embed': dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', power=0.9, by_epoch=False, begin=500, end=40000, eta_min=0.0),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

