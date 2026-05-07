_base_ = ['../mask2former_finetune_synthetic.py']

model = dict(
    decode_head=dict(
        loss_cls=dict(
            class_weight=[1.0] * 10 + [0.8],
        ),
    ),
)
