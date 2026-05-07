"""Mask2Former Stage-2C: real fine-tune from Stage-2A synth-pretrained ckpt.

Same train/val/test composition as mask2former_finetune_real.py
(real_v2/train + opensrc/train; combined val/test). LR halved, iters reduced,
val cadence tightened to catch the peak before any synth-domain drift.

IMPORTANT: set BLV_STAGE2A_CKPT env var to the Stage-2A best ckpt path before
launching, or override `load_from` via --cfg-options.
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
real_root = _os.path.join(_blv_root, 'data', 'real_v2')
opensrc_root = _os.path.join(_blv_root, 'data', 'opensrc')

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
    excluded_class_indices=[10],  # turnstile: zero GT in real_v2/opensrc val+test
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
    excluded_class_indices=[10],  # turnstile: zero GT in real_v2/opensrc val+test
)

# Half of Stage-1 LR (1e-5 → 5e-6) — model already aligned to BLV classes.
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-6, weight_decay=0.05),
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

# 7k iters ≈ 15% of Stage-1's 45k. Short tail to lock in real-domain peak.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=200),
    dict(type='PolyLR', power=0.9, by_epoch=False, begin=200, end=7000, eta_min=0.0),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=7000, val_begin=1000, val_interval=1000)
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
        interval=1000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=500),
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

# Stage-2A best ckpt — set BLV_STAGE2A_CKPT or override via --cfg-options.
load_from = _os.environ.get(
    'BLV_STAGE2A_CKPT',
    _os.path.join(_blv_root, 'checkpoints', 'stage2a_best', 'mask2former_synth_best.pth'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='blv-seg',
            name='stage-2c-real-from-synth-mask2former',
            tags=['stage-2c', 'real-finetune-from-synth', 'mask2former', 'v2-schema'],
        ),
    ),
]

visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
