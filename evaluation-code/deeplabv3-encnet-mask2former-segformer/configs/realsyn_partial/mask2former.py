#configs/my_model/[project-name]/realsyn_partial/

_base_ = ['../../../mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py']

custom_imports = dict(
    imports=[
        'mmseg.evaluation.fg_iou_metric',
        'mmseg.datasets.my_mmseg_dataset',
    ],
    allow_failed_imports=False,
)

dataset_type = 'MyMMSegDataset'

data_roots = [
    'PATH_TO_REAL_DATA',
    'PATH_TO_CURATED_OR_SYN_DATA',
]

crop_size = (640, 640)

NUM_CLASSES = 12
BG_INDEX = 0
EXCLUDED_FROM_FG_MEAN = [BG_INDEX]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2560, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        cat_max_ratio=0.75,
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(2560, 640),
        keep_ratio=True,
    ),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]


def build_concat_dataset(split, pipeline):
    return dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path=f'img_dir/{split}',
                    seg_map_path=f'ann_dir/{split}',
                ),
                pipeline=pipeline,
            )
            for data_root in data_roots
        ],
    )


train_dataloader = dict(
    batch_size=6,  # per GPU. 4 GPUs -> effective batch 24
    num_workers=8,
    persistent_workers=True,
    dataset=build_concat_dataset('train', train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root='PATH_TO_REAL_TEST_DATA',
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val',
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root='PATH_TO_REAL_TEST_DATA',
        data_prefix=dict(
            img_path='img_dir/test',
            seg_map_path='ann_dir/test',
        ),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type='FgIoUMetric',
    iou_metrics=['mIoU', 'mFscore'],
    excluded_class_indices=EXCLUDED_FROM_FG_MEAN,
    ignore_index=255,
)

test_evaluator = dict(
    type='FgIoUMetric',
    iou_metrics=['mIoU', 'mFscore'],
    excluded_class_indices=EXCLUDED_FROM_FG_MEAN,
    ignore_index=255,
)

# 12 semantic classes + 1 no-object class for Mask2Former classification loss
mask2former_class_weight = [
    0.10,  # background
    1.00,  # elevator
    1.00,  # elevator_button
    1.00,  # door_button
    1.00,  # crosswalk
    1.00,  # pedestrian_signal
    1.00,  # aps_button
    1.00,  # bus_stop
    1.00,  # bus_stop_sign
    1.00,  # handrail
    1.00,  # escalator
    1.00,  # turnstile
    0.05,  # no-object
]

model = dict(
    decode_head=dict(
        num_classes=NUM_CLASSES,
        ignore_index=255,
        loss_cls=dict(
            _delete_=True,
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=mask2former_class_weight,
        ),
    ),
)

# AMP disabled for Hungarian matcher stability
# LR partial scaling: 1e-5 -> 1.5e-5
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'query_embed': dict(decay_mult=0.0),
            'query_feat': dict(decay_mult=0.0),
            'level_embed': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500,
    ),
    dict(
        type='PolyLR',
        power=1.0,
        eta_min=0.0,
        by_epoch=False,
        begin=1500,
        end=80000,
    ),
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=80000,
    val_begin=1500,
    val_interval=1500,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

log_processor = dict(by_epoch=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=200,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1500,
        save_best='fg_mIoU',
        rule='greater',
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=2000,
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
