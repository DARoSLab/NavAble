import os.path as osp

import mmcv
from mmengine.fileio import get
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class IntervalSegVisualizationHook(SegVisualizationHook):
    """SegVisualizationHook subclass that respects ``interval`` in test mode.

    Upstream mmseg 1.2.2 ignores ``interval`` in ``after_test_iter`` and
    visualizes every test image. This subclass adds a modulo check so only
    every ``interval``-th test image is rendered.
    """

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        if self.draw is False:
            return

        for data_sample in outputs:
            self._test_index += 1
            if (self._test_index - 1) % self.interval != 0:
                continue

            img_path = data_sample.img_path
            window_name = f'test_{osp.basename(img_path)}'
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index,
            )
