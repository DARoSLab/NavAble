# configs/navable/real_only/encnet.py
_base_ = ['../../../encnet/encnet_r101-d8_4xb4-80k_ade20k-512x512.py']

custom_imports = dict(
    imports=['mmseg.evaluation.fg_iou_metric', 'mmseg.datasets.my_mmseg_dataset'],
    allow_failed_imports=False,
)

dataset_type = 'MyMMSegDataset'
data_roots = [
    'PATH_TO_REAL_DATA',
    'PATH_TO_CURATED_OR_SYN_DATA',
]

crop_size = (512, 512)

NUM_CLASSES = 12
BG_INDEX = 0
EXCLUDED_FROM_FG_MEAN = [BG_INDEX]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
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
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    dataset=build_concat_dataset('train', train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=build_concat_dataset('val', test_pipeline),
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

class_weight = [
    0.10, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
]

model = dict(
    decode_head=dict(
        num_classes=NUM_CLASSES,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=class_weight,
        ),
    ),
    auxiliary_head=dict(
        num_classes=NUM_CLASSES,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=class_weight,
        ),
    ),
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=300),
    dict(type='PolyLR', power=0.9, eta_min=1e-4, by_epoch=False, begin=300, end=16000),
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=16000,
    val_begin=400,
    val_interval=400,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

log_processor = dict(by_epoch=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=400,
        save_best='fg_mIoU',
        rule='greater',
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=200),
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
