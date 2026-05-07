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
dataset_type = 'BLVDataset'
_blv_root = _os.environ['BLV_PROJECT_ROOT'] if 'BLV_PROJECT_ROOT' in _os.environ else _os.path.abspath(_os.path.join(_os.getcwd(), '..', '..'))
data_root = _os.environ.get(
    'BLV_REAL_TESTSET_ROOT',
    _os.path.join(_blv_root, 'data', 'real-temp-test-uploaded-soowan', 'mmseg_dataset_real_testset'),
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=(640, 640), max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        ann_file='splits/test.txt',
        pipeline=test_pipeline,
    ),
)

test_evaluator = dict(
    _delete_=True,
    type='BLVMetric',
    iou_metrics=['mIoU', 'mAP50-95', 'Prec', 'Rec'],
    zero_shot_remap=False,
    num_classes=10,
    mask_fg_conf_threshold=0.0,
    ignore_index=255,
    gt_label_map={
        0: 0,  # elevator
        1: 1,  # elevator_button
        2: 2,  # door_button
        3: 4,  # crosswalk
        4: 6,  # pedestrian_signal
        5: 7,  # push_button
        6: 8,  # bus_stop
        7: 9,  # bus_stop_sign
        8: 3,  # door
    },
    output_metrics_path=None,
)

model = dict(
    decode_head=dict(
        type='BLVSideAdapterCLIPHead',
        num_classes=10,
        query_fg_threshold=0.1,
        bg_threshold=0.5,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_cls_ce',
                loss_weight=2.0,
                class_weight=[1.0] * 10 + [0.1],
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
            'elevator', 'elevator button', 'door button', 'door',
            'crosswalk', 'traffic signal', 'pedestrian signal',
            'push button', 'bus stop', 'bus stop sign',
        ),
    ),
)

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50),
)

vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
