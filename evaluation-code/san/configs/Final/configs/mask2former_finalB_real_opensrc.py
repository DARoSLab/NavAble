"""Mask2Former Final-B: real + opensrc training, real-only eval. Paper-canonical baseline.

- ADE20k pretrained init.
- Train: ConcatDataset(real_final + opensrc_final) train.
- Val:   data/real_final/val (REAL only — best ckpt by real-domain mIoU).
- Test:  data/real_final/test.
- 45,000 iters at batch 4 (Stage-1 baseline iters).
"""

crop_size = (640, 640)

_base_ = [
    '../../checkpoints/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py',
]

import os as _os

custom_imports = dict(
    imports=[
        'blv_pipeline.mmseg_plugins.datasets.blv_dataset',
        'blv_pipeline.mmseg_plugins.evaluation.blv_metric',
        'blv_pipeline.mmseg_plugins.decode_heads.blv_mask2former_head',
    ],
    allow_failed_imports=False,
)

default_scope = 'mmseg'
dataset_type = 'BLVDatasetV2Fg'
_blv_root = _os.environ['BLV_PROJECT_ROOT'] if 'BLV_PROJECT_ROOT' in _os.environ else _os.path.abspath(_os.path.join(_os.getcwd(), '..', '..'))
real_root = _os.path.join(_blv_root, 'data', 'real_final')
opensrc_root = _os.path.join(_blv_root, 'data', 'opensrc_final')

NUM_FG = 11

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1280, 640), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(type=dataset_type, data_root=real_root,
                 data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                 pipeline=train_pipeline),
            dict(type=dataset_type, data_root=opensrc_root,
                 data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                 pipeline=train_pipeline),
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
        type=dataset_type,
        data_root=real_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=real_root,
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        pipeline=test_pipeline,
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

optim_wrapper = dict(
    _delete_=True,
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
    dict(type='PolyLR', power=0.9, by_epoch=False, begin=500, end=45000, eta_min=0.0),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=45000, val_begin=3000, val_interval=3000)
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
        interval=3000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=2000),
)

model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(
        type='BLVMask2FormerHead',
        num_classes=NUM_FG,
        query_fg_threshold=0.1,
        bg_threshold=0.5,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * NUM_FG + [0.1],
        ),
    ),
)

load_from = _os.path.join(_blv_root, 'checkpoints', 'pretrained', 'mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.pth')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='blv-seg-final',
            name='mask2former-final-b-real-opensrc',
            tags=['final', 'configB', 'mask2former'],
        ),
    ),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
