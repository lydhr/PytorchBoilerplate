import logging
import torch
import torch.nn as nn
from utils import utils, signal_utils

# A logger for this file
log = logging.getLogger(__name__)

def get_loss(y_pred, y, loss_cfg):
    match loss_cfg.type:
        case "CrossEntropy":
            return crossEntropy_loss(y_pred, y)
        case "MSE":
            return mse_loss(y_pred, y)
        case default:
            raise ValueError("Invalid SignalType: {}".format(loss_cfg.type))

def crossEntropy_loss(y_pred, y):
    '''
        y_pred, y: (batch_size, 2)
        return: (batch_size,)
    '''
    loss = nn.CrossEntropyLoss()
    return loss(y_pred, y)


def mse_loss(y_pred, y):
    '''
        y_pred, y: (batch_size, 2)
        return: (batch_size,)
    '''
    loss = nn.MSELoss()
    return loss(y_pred, y)