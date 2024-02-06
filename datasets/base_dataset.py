import numpy as np
import logging
import torch
from torch.utils.data import Dataset

from utils import signal_utils, file_utils

# A logger for this file
log = logging.getLogger(__name__)

class BaseDataset(Dataset):

    def __init__(self, fnames, signal_cfg):
        self.parse_config(signal_cfg)
        self.x = self.get_data(fnames)
        self.y = self.get_label()
        self.shape() #(512,), (xxx, 512)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.y)

    def parse_config(self, cfg):
        self.FS = cfg.fs
        self.WINDOW_SIZE = cfg.window_size

    def get_data(self, files_path):
        if len(files_path) < 1:
            raise ValueError('Dataset input files_path is empty.')
        x = self.load_data_files(files_path=files_path, size=self.WINDOW_SIZE)
        return self.set_precision(x)

    def get_label(self):
        y = self.get_label_pseudo(self.x.shape)
        return self.set_precision(y)

    def set_precision(self, data, precision=np.float32):
        #default is model.double()
        if type(data) != precision:
            return data.astype(precision)
        else:
            return data

    def shape(self):
        d_shape = (self.x.shape)
        log.info("{} x.shape = {}".format(type(self).__name__, d_shape))
        return d_shape

    def load_data_files(self, files_path, size):
        x = []
        data_list = file_utils.read_wav(files_path)
        for fs, data  in data_list:
            x.append(data) # data = (n_sample, n_channel) or (n_sample, ) when n_channel = 1
        x = np.concatenate(x)
        n = x.shape[0]//size
        return x[:size*n].reshape(n, size)
    
