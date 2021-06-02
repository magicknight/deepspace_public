from numpy.lib.index_tricks import _diag_indices_from
from torch import nn
import torch


def Linear(in_features, out_features, activation=None, batchnorm=False):
    layers = [nn.Linear(in_features, out_features)]

    if activation is not None:
        layers += [activation]
    if batchnorm:
        layers += [nn.BatchNorm1d(out_features)]

    return nn.Sequential(*layers)


class MLP(nn.Module):
    """this mlp includes two parts: wave mlp and detector mlp
    wave mlp is a shared parameters mlp with input size as the waveform size. detector mlp is a normal mlp with input size the number of detectors

    Args:
        nn (torch nn): torch nn
    """

    def __init__(self, input_shape, wave_mlp_sizes, det_mlp_sizes, *, activation=None):
        """init

        Args:
            input_shape (list): two dimension, [detector dim, wave dim]
            wave_mlp_sizes (list): a list of dim for each layer
            det_mlp_sizes (list): a list of dim for each layer for detector mlp
            activation (activation function, optional): activation function. only for last layer. Defaults to None.
        """
        super().__init__()
        self.input_shape = input_shape
        # wave mlp
        wave_dims = list(zip([input_shape[1], *wave_mlp_sizes], wave_mlp_sizes))
        self.wave_mlp = nn.Sequential(
            *[Linear(*features, activation=nn.GELU(), batchnorm=False) for features in wave_dims[:-1]],
            Linear(*wave_dims[-1], activation=activation, batchnorm=False)
        )

        # # detector mlp
        det_dims = list(zip([input_shape[0], *det_mlp_sizes], det_mlp_sizes))
        self.det_mlp = nn.Sequential(
            *[Linear(*features, activation=nn.GELU(), batchnorm=False) for features in det_dims[:-1]],
            Linear(*det_dims[-1], activation=activation, batchnorm=False)
        )

    def wave(self, x):
        # [batch, column, row] => [column, batch, row]? no, we will make it like [batch * column, row]
        y = self.wave_mlp(x)
        return y

    def detector(self, index, x):
        x = torch.stack((index, x.squeeze(dim=-1)), dim=-1)
        x = x.view(x.shape[0], -1)
        y = self.det_mlp(x)
        return y

    def forward(self, index, x):
        """
        """
        # print(shape)
        # index = index[:, :shape]
        # x = x[:, :shape, :]
        return self.detector(index, self.wave(x))
