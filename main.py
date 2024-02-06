# import matplotlib.pyplot as plt
import os
from itertools import combinations
import hydra
from omegaconf import DictConfig
import logging

import torch
from torch.utils.data import DataLoader, ConcatDataset

from datasets import *
from models import *
from utils import utils
import utils.dataset_utils as ds_utils
from trainer.trainer import Trainer

if torch.cuda.is_available():
    # ensure CUDA deterministic
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" # will increase library footprint in GPU memory by approximately 24MiB
    torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.benchmark = False

utils.init_hydra()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    utils.set_seed(cfg.seed)

    model = globals()[cfg.model.arch]()

    # prepare datasets
    piano_train_ds, piano_test_ds = ds_utils.get_split_datasets(ds_class=PianoDataset, cfg=cfg)
    podcast_train_ds, podcast_test_ds = ds_utils.get_split_datasets(ds_class=PodcastDataset, cfg=cfg)
    
    train_ds, val_ds = ds_utils.random_split_dataset(ConcatDataset([piano_train_ds, podcast_train_ds]), split_size=0.2)
    
    datasets = {"train": train_ds,
                "validation": val_ds,
                "test": ConcatDataset([piano_test_ds, podcast_test_ds])}
    log.info("datasets = {}".format([f'{k}: {v.__len__()}' for k, v in datasets.items()]))

    # train and test
    trainer = Trainer(model=model, datasets=datasets, model_cfg=cfg.model)
    trainer.train()
    trainer.predict()



if __name__ == "__main__":
    main()
