_base_ = ['../san_finetune_synthetic.py']

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_cls_ce',
                loss_weight=2.0,
                class_weight=[1.0] * 10 + [0.4],
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
)
