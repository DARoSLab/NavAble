"""SAN zero-shot eval on real_v2/test + opensrc/test (no BLV fine-tuning).

Uses the COCO-stuff pretrained SAN checkpoint with BLV class text prompts.
The CLIP text encoder queries for BLV classes directly — no weight updates.
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=(640, 640), max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

# Only test_dataloader matters for zero-shot eval — train/val kept minimal
train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=real_root,
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        pipeline=test_pipeline,
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
    excluded_class_indices=[10],  # turnstile: zero GT in real_v2/opensrc test
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
    excluded_class_indices=[10],  # turnstile: zero GT in real_v2/opensrc test
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

# COCO-stuff pretrained weights — no BLV fine-tuning
load_from = _os.path.join(_blv_root, 'checkpoints', 'pretrained', 'san-vit-b16_20230906-fd0a7684.pth')

test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')
train_cfg = dict(type='IterBasedTrainLoop', max_iters=1, val_begin=0, val_interval=1)
log_processor = dict(by_epoch=False)

default_hooks = dict(
    _delete_=True,
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
