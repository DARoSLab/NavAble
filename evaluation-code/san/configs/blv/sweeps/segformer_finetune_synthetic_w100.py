_base_ = ['../segformer_finetune_synthetic.py']

model = dict(
    decode_head=dict(
        loss_decode=dict(
            class_weight=[1.0] * 10 + [1.0],
        ),
    ),
)
