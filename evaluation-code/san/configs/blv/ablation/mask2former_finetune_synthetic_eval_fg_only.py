_base_ = ['../mask2former_finetune_synthetic.py']

# A/B evaluator variant: ignore extra background channel(s) at eval time,
# score only foreground logits with a low confidence gate.
val_evaluator = dict(
    segformer_extra_channel_fg_only=True,
    segformer_conf_threshold=0.05,
    mask_fg_conf_threshold=0.0,
)

test_evaluator = dict(
    segformer_extra_channel_fg_only=True,
    segformer_conf_threshold=0.05,
    mask_fg_conf_threshold=0.0,
)
