import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
from utils import utils

# A logger for this file
log = logging.getLogger(__name__)


def generate_sine(f, fs, size, max_amplitude):
    t = np.arange(0, size, 1)/fs
    phase = np.pi * 2 * f * t
    samples = np.round(np.sin(phase) * max_amplitude) #18k * 512/48k = 192 periods

    log_sim_samples_stats(samples, fs, f"sine {f}hz")
    return samples

def get_FFT(x):
    '''
        y: torch.tensor or numpy.ndarry, (n, win_size) or (ws, )
        return: torch.tensor (n, win_size//2+1) or (ws//2+1)
    '''
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)
    return torch.abs(torch.fft.rfft(x))

def plot_FFT(x, cfg, save_path=None):
    '''
        x: numpy array, (n, window_size)
        save_path: str or None
    '''
    plt.figure()
    freq = get_FFT(x)
    freq = freq.detach().cpu().numpy()
    for i in range(1): # temporarily only plot the first window
        plt.plot(freq[i])
    
    plt.title(save_path)
    # save and show
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = "tight")
        log.debug('Saved image in {}'.format(save_path))
     
    if cfg.show:
        plt.show(block=False)
        plt.pause(0.001)

