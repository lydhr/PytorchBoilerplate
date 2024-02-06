import numpy as np
import random
from datetime import datetime
import time
import logging
import torch
import hydra
from omegaconf import OmegaConf
import os

# A logger for this file
log = logging.getLogger(__name__)

def init_hydra():
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("floor", lambda x: int(x)) # int is already a premitive type, so we use floor.

def get_hydra_output_dir():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    dir_path = hydra_cfg['runtime']['output_dir']
    return dir_path

def get_hydra_output_path(relative_path):
    dir_path = get_hydra_output_dir()
    path = os.path.join(dir_path, relative_path)
    return path

def getTimestamp():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    return dt_string

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def timeit(func):
    def inner(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        log.info("Timeit {}() = {}s".format(func.__name__, time.time() - start))
        return res
    return inner

def check_range(arrs, left=0, right=1):
    # arrs: list of torch arrays, or 2d tensor
    if not torch.is_tensor(arrs):
        arrs = torch.stack(arrs)
    l = torch.min(arrs, axis=1).values
    r = torch.max(arrs, axis=1).values
    if not (all(left<=l) and all(r<=right)):
        print(f'{arrs.shape} out of range {l}-{r}')

def log_accuracy(y_pred_y_arr):
    """
        y_pred_y_arr: (n, 2, 2)
    """
    y_pred_y_arr = np.argmax(y_pred_y_arr, axis=2)
    y_pred = y_pred_y_arr[:, 0]
    y = y_pred_y_arr[:, 1]
    correct = (y == y_pred).sum()

    log.info('accuracy {}/{}'.format(correct, len(y)))
