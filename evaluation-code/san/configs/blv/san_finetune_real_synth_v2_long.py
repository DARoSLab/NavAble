"""SAN Stage-2C-LONG: real fine-tune from Stage-2A synth-pretrained ckpt.

- iters bumped 4.5k → 30k (= Stage-1)
- LR restored 5e-5 → 1e-4 (= Stage-1)
- batch 16 → 24 (better L40s utilization, ~35GB peak est.)
- val cadence 500 → 2k
"""

crop_size = (640, 640)

_base_ = [
    '../../checkpoints/san-vit-b16_coco-stuff164k-640x640.py',
]

import os as _os

custom_imports = dict(
    imports=[
        'blv_pipeline.mmseg_plugins.datasets.blv_dataset',
        'blv_pipeline.mmseg_plugins.evaluation.blv_metric',
        'blv_pipeline.mmseg_plugins.decode_heads.blv_san_head',
    ],
    allow_failed_imports=False,
)

default_scope = 'mmseg'
dataset_type = 'BLVDatasetV2Fg'
_blv_root = _os.environ['BLV_PROJECT_ROOT'] if 'BLV_PROJECT_ROOT' in _os.environ else _os.path.abspath(_os.path.join(_os.getcwd(), '..', '..'))
real_root = _os.path.join(_blv_root, 'data', 'real_v2')
opensrc_root = _os.path.join(_blv_root, 'data', 'opensrc')

NUM_FG = 11

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        resize_type='ResizeShortestEdge',
        scales=[320, 384, 448, 512, 576, 640],
        max_size=2560,
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=(640, 640), max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=24,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=real_root,
                data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                data_root=opensrc_root,
                data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                pipeline=train_pipeline,
            ),
        ],
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=real_root,
                data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                data_root=opensrc_root,
                data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                pipeline=test_pipeline,
            ),
        ],
    ),
)

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=real_root,
                data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                data_root=opensrc_root,
                data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
                pipeline=test_pipeline,
            ),
        ],
    ),
)

val_evaluator = dict(
    _delete_=True,
    type='BLVMetric',
    iou_metrics=['mIoU', 'mAP50-95', 'Prec', 'Rec'],
    zero_shot_remap=False,
    num_classes=NUM_FG,
    mask_fg_conf_threshold=0.0,
    ignore_index=255,
    output_metrics_path=None,
    excluded_class_indices=[10],
)

test_evaluator = dict(
    _delete_=True,
    type='BLVMetric',
    iou_metrics=['mIoU', 'mAP50-95', 'Prec', 'Rec'],
    zero_shot_remap=False,
    num_classes=NUM_FG,
    mask_fg_conf_threshold=0.0,
    ignore_index=255,
    output_metrics_path=None,
    excluded_class_indices=[10],
)

# Stage-1 SAN LR (1e-4).
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'norm': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'cls_token': dict(decay_mult=0.0),
        },
    ),
)

# 30k iters = Stage-1 length.
param_scheduler = [
    dict(type='PolyLR', power=1.0, by_epoch=False, begin=0, end=30000, eta_min=0.0),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_begin=2000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(by_epoch=False)

default_hooks = dict(
    _delete_=True,
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1000),
)

model = dict(
    decode_head=dict(
        type='BLVSideAdapterCLIPHead',
        num_classes=NUM_FG,
        query_fg_threshold=0.1,
        bg_threshold=0.5,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_cls_ce',
                loss_weight=2.0,
                class_weight=[1.0] * NUM_FG + [0.1],
            ),
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_name='loss_mask_ce',
                loss_weight=5.0,
            ),
            dict(
                type='DiceLoss',
                naive_dice=True,
                eps=1,
                loss_name='loss_mask_dice',
                loss_weight=5.0,
                ignore_index=None,
            ),
        ],
    ),
    text_encoder=dict(
        dataset_name=None,
        vocabulary=(
            'elevator', 'elevator button', 'door button',
            'crosswalk', 'pedestrian signal',
            'accessible pedestrian signal button',
            'bus stop', 'bus stop sign',
            'handrail', 'escalator', 'turnstile',
        ),
    ),
)

load_from = _os.environ.get(
    'BLV_STAGE2A_CKPT',
    _os.path.join(_blv_root, 'checkpoints', 'stage2a_best', 'san_synth_best.pth'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='blv-seg',
            name='stage-2c-long-real-from-synth-san',
            tags=['stage-2c-long', 'real-finetune-from-synth', 'san', 'v2-schema'],
        ),
    ),
]

visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
