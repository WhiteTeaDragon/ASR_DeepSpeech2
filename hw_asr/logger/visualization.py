from .tensorboard import TensorboardWriter
from .wandb import WanDBWriter


def get_visualizer(config, logger, vis_type):
    if vis_type == "tensorboard":
        return TensorboardWriter(config.log_dir, logger, True)

    if vis_type == 'wandb':
        return WanDBWriter(config, logger)

    return None
