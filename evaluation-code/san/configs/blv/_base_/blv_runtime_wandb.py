# ---------------------------------------------------------------------------
# Visualization backend: LocalVisBackend + WandbVisBackend (PERMANENT DEFAULT)
#
# draw=True in SegVisualizationHook is the correct, permanent setting.
# It enables val-time segmentation overlays to be logged to W&B so that
# val/* metrics and sample image grids appear correctly in the W&B dashboard.
#
# Root cause of past W&B logging crashes (DO NOT revert to draw=False as fix):
#   The crash "numpy.dtype size changed, may indicate binary incompatibility"
#   was an ABI mismatch between numpy and pandas, triggered when wandb tried
#   to import pandas during image logging.  The fix is in environment.yml:
#   pin numpy>=2.2,<2.3 and pandas>=2.2,<2.3 from conda-forge together so
#   their C extensions are built against the same numpy headers.
#
# draw=False is available only as a temporary survival fallback passed via
# --no-wandb on the CLI (tools/blv/train.sh / slurm_train.sh).  It should
# NOT be committed to any config file.
# ---------------------------------------------------------------------------
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(project='blv-seg', tags=['finetune', 'mask2former']),
    ),
]

visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # draw=True: permanent setting — see comment block above before changing.
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1000),
)
