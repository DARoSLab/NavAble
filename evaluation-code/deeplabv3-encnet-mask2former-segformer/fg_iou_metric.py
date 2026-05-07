#~/mmsegmentation/mmseg/evaluation/
from collections import OrderedDict

import numpy as np
from prettytable import PrettyTable

from mmengine.logging import MMLogger
from mmseg.evaluation import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class FgIoUMetric(IoUMetric):

    def __init__(self, excluded_class_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.excluded_class_indices = list(excluded_class_indices or [])

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()

        if getattr(self, 'format_only', False):
            logger.info(f'results are saved to {self.output_dir}')
            return OrderedDict()

        # Batch results
        results = tuple(zip(*results))
        assert len(results) == 4, (
            f'Expected 4 tuple components from IoUMetric.process, got {len(results)}'
        )
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])

        # per-class basic metric calculation
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = list(self.dataset_meta['classes'])
        num_classes = len(class_names)

        # fg mask
        fg_mask = np.array(
            [i not in self.excluded_class_indices for i in range(num_classes)],
            dtype=bool,
        )

        out = {}
        for key, val in ret_metrics.items():
            if key == 'aAcc':
                out[key] = float(np.round(np.asarray(val) * 100, 2))
                continue
            arr = np.asarray(val) * 100
            out[f'm{key}'] = float(np.round(np.nanmean(arr), 2))
            if fg_mask.any():
                out[f'fg_m{key}'] = float(np.round(np.nanmean(arr[fg_mask]), 2))

        ret_metrics_class = OrderedDict()
        ret_metrics_class['Class'] = class_names
        for key, val in ret_metrics.items():
            if key == 'aAcc':
                continue
            ret_metrics_class[key] = np.round(np.asarray(val) * 100, 2).tolist()

        table = PrettyTable()
        for col, vals in ret_metrics_class.items():
            table.add_column(col, vals)

        excluded_names = [class_names[i] for i in self.excluded_class_indices
                          if 0 <= i < num_classes]
        logger.info(
            f'per class results (excluded from fg means: {excluded_names}):'
        )
        logger.info('\n' + table.get_string())

        return out