import numpy as np
import os
import logging

from utils import file_utils
from .base_dataset import BaseDataset

# A logger for this file
log = logging.getLogger(__name__)

class PianoDataset(BaseDataset):
    CONFIG_NAME = "piano"
    # DATA_FORMAT = "wav"
    # N_CHANNEL = 1


    def get_files_path(data_cfg, keyword=".wav"):
        """Return files path sorted by the number in the filename. """
        path = data_cfg["datasets"][PianoDataset.CONFIG_NAME]["folder_path"] # Get hydra config.
        files = [os.path.join(path, f) for f in os.listdir(path) if keyword in f]
        files.sort()
        log.info('load {} files in {}'.format(len(files), path))
        return files

    def get_label_pseudo(self, x_shape):
        return np.array([[1, 0]] * x_shape[0]) # [0, 1] = podcast, [1, 0] = piano