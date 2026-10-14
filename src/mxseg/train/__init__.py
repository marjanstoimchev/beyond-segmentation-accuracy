"""Training baselines with PyTorch Lightning. One strict config, one runner; models plug in as LightningModules.

    mxseg train -c configs/train_cellpose.yaml          # fine-tune Cellpose-SAM v2 on our labels -> runs/<name>/weights.pt
    mxseg eval  -c configs/slide-0235.yaml --baseline cellpose --param variant=runs/<name>/weights.pt --name cellpose_ft
"""
from .config import TrainConfig, load_train_config
from .run import train

__all__ = ["TrainConfig", "load_train_config", "train"]
