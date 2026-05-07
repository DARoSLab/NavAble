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
dataset_type = 'BLVDataset'
_blv_root = _os.environ['BLV_PROJECT_ROOT'] if 'BLV_PROJECT_ROOT' in _os.environ else _os.path.abspath(_os.path.join(_os.getcwd(), '..', '..'))
data_root = _os.environ.get(
    'BLV_REAL_TESTSET_ROOT',
    _os.path.join(_blv_root, 'data', 'real-temp-test-uploaded-soowan', 'mmseg_dataset_real_testset'),
)

crop_size = (1024, 1024)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2560, 640), keep_ratio=True),
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
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(
        type='BLVMask2FormerHead',
        num_classes=10,
        query_fg_threshold=0.1,
        bg_threshold=0.5,
    ),
)

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50),
)

vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
