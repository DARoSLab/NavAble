from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class MyMMSegDataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            'background',
            'elevator',
            'elevator_button',
            'door_button',
            'crosswalk',
            'pedestrian_signal',
            'aps_button',
            'bus_stop',
            'bus_stop_sign',
            'handrail',
            'escalator',
            'turnstile',
        ),
        palette=[
            [0, 0, 0],        # background
            [128, 0, 0],      # elevator
            [0, 128, 0],      # elevator_button
            [128, 128, 0],    # door_button
            [0, 0, 128],      # crosswalk
            [128, 0, 128],    # pedestrian_signal
            [0, 128, 128],    # aps_button
            [128, 128, 128],  # bus_stop
            [64, 0, 0],       # bus_stop_sign
            [0, 64, 0],       # handrail
            [64, 64, 0],      # escalator
            [0, 0, 64],       # turnstile
        ]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs
        )