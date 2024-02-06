import torch
import logging

log = logging.getLogger(__name__)

def random_split_dataset(dataset, split_size=0.2):
    """
        return: 
            ds_a with n * (1 - split_size) samples
            ds_b with n * split_size samples
    """
    size_b = int(split_size * len(dataset))
    size_a = len(dataset) - size_b
    datasets = {}
    ds_a, ds_b = torch.utils.data.random_split(dataset, [size_a, size_b])
    
    return ds_a, ds_b

def get_full_dataset(ds_class, cfg, keyword=".wav"):
    files_path = ds_class.get_files_path(data_cfg=cfg.data, keyword=keyword)
    ds = ds_class(fnames=files_path, signal_cfg=cfg.signal)
    return ds

def get_split_datasets(ds_class, cfg, n_test_file=1, keyword=".wav"):
    files_path = files_path = ds_class.get_files_path(data_cfg=cfg.data, keyword=keyword)

    train_ds = ds_class(fnames=files_path[:-n_test_file], signal_cfg=cfg.signal)
    test_ds = ds_class(fnames=files_path[-n_test_file:], signal_cfg=cfg.signal)
    return train_ds, test_ds