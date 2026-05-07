from mmengine.registry import VISBACKENDS
from mmengine.visualization import WandbVisBackend


@VISBACKENDS.register_module()
class WandbMetricsOnlyBackend(WandbVisBackend):
    def add_image(self, name, image, step=0, **kwargs):
        return

    def add_images(self, name, image, step=0, **kwargs):
        return
