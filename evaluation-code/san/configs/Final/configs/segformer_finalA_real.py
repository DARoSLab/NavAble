"""SegFormer Final-A: real-only training, real-only eval. Paper-canonical baseline.

- ADE20k pretrained init.
- Train: data/real_final/train (3,703 imgs).
- Val:   data/real_final/val (396 imgs) — best-ckpt selection under honest BLVMetric.
- Test:  data/real_final/test (1,482 imgs) — evaluated separately via slurm_eval_final.
- 12,000 iters at batch 10 (~32 epochs over 3,703 imgs).
"""

crop_size = (640, 640)

_base_ = [
    '../../checkpoints/segformer_mit-b5_8xb2-160k_ade20k-640x640.py',
]

import os as _os

custom_imports = dict(
    imports=[
        'blv_pipeline.mmseg_plugins.datasets.blv_dataset',
        'blv_pipeline.mmseg_plugins.datasets.remap_seg_label',
        'blv_pipeline.mmseg_plugins.evaluation.blv_metric',
    ],
    allow_failed_imports=False,
)

default_scope = 'mmseg'
dataset_type = 'BLVDatasetV2Fg'
_blv_root = _os.environ['BLV_PROJECT_ROOT'] if 'BLV_PROJECT_ROOT' in _os.environ else _os.path.abspath(_os.path.join(_os.getcwd(), '..', '..'))
real_root = _os.path.join(_blv_root, 'data', 'real_final')

NUM_FG = 11
BG_CLASS_ID = NUM_FG

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RemapSegLabel', src_label=255, dst_label=BG_CLASS_ID),
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
    batch_size=10,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=real_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline,
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
    ignore_index=255,
    segformer_conf_threshold=0.05,
    mask_fg_conf_threshold=0.0,
    segformer_extra_channel_fg_only=True,
    output_metrics_path=None,
    excluded_class_indices=[10],
)

test_evaluator = dict(
    _delete_=True,
    type='BLVMetric',
    iou_metrics=['mIoU', 'mAP50-95', 'Prec', 'Rec'],
    zero_shot_remap=False,
    num_classes=NUM_FG,
    ignore_index=255,
    segformer_conf_threshold=0.05,
    mask_fg_conf_threshold=0.0,
    segformer_extra_channel_fg_only=True,
    output_metrics_path=None,
    excluded_class_indices=[10],
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'norm': dict(decay_mult=0.0),
            'pos_block': dict(decay_mult=0.0),
        },
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=300),
    dict(type='PolyLR', power=1.0, by_epoch=False, begin=300, end=12000, eta_min=0.0),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=12000, val_begin=500, val_interval=500)
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
        interval=500,
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
        num_classes=NUM_FG + 1,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0] * NUM_FG + [0.15],
        ),
    ),
)

load_from = _os.path.join(_blv_root, 'checkpoints', 'pretrained', 'segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='blv-seg-final',
            name='segformer-final-a-real',
            tags=['final', 'configA', 'segformer'],
        ),
    ),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
