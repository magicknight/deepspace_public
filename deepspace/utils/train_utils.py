import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import math

from commontools.setup import config, logger
from deepspace.utils.misc import print_cuda_statistics

"""
Learning rate adjustment used for CondenseNet model training
"""


def adjust_learning_rate(optimizer, epoch, config, batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = config.max_epoch * nBatch
        T_cur = (epoch % config.max_epoch) * nBatch + batch
        lr = 0.5 * config.learning_rate * (1 + math.cos(math.pi * T_cur / T_total))
    else:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = config.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_device():
    use_cuda = config.deepspace.device == 'gpu'
    is_cuda = torch.cuda.is_available()
    if is_cuda and not use_cuda:
        logger.warn("You have a CUDA device, so you should probably enable CUDA!")
    cuda = is_cuda and use_cuda
    # set the manual seed for torch
    if cuda:
        torch.cuda.manual_seed_all(config.deepspace.seed)
        device = torch.device("cuda")
        torch.cuda.set_device(config.deepspace.gpu_device)
        logger.info("Program will run on *****GPU-CUDA*****, device $1".format(config.deepspace.gpu_device))
        print_cuda_statistics()
    else:
        device = torch.device("cpu")
        torch.manual_seed(config.deepspace.seed)
        logger.info("Program will run on *****CPU*****\n")
    return device
