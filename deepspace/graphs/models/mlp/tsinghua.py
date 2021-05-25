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
        # detector mlp
        det_dims = list(zip([input_shape[0], *det_mlp_sizes], det_mlp_sizes))
        self.det_mlp = nn.Sequential(
            *[Linear(*features, activation=nn.GELU(), batchnorm=False) for features in det_dims[:-1]],
            Linear(*det_dims[-1], activation=activation, batchnorm=False)
        )

    def wave(self, x):
        # [batch, column, row] => [column, batch, row]? no, we will make it like [batch * column, row]
        shape = x.shape[1]
        x = x.view(-1, self.input_shape[1])
        y = self.wave_mlp(x)
        y = y.view(-1, shape)
        return y

    def detector(self, index, x):
        y = torch.zeros(x.shape[0], self.input_shape[0] + 1, requires_grad=True).to(x.device)
        # for each_index, each_x, each_y in zip(index, x, y):
        #     each_y[each_index] = each_x
        y = y.scatter(1, index.type(torch.int64), x)
        # since index = 0 are addition data, drop it.
        y = y[:, 1:]
        y = self.det_mlp(y)
        return y

    def forward(self, index, x):
        """
        model = MLP([43212, 1000], [ 256, 64, 1 ], [ 256, 64, 1 ], activation=nn.Sigmoid())
        summary(model, (1, *config.deepspace.shape))
        x = torch.randn(2, *config.deepspace.shape)
        y = model(x)
        tuple(y.shape)
        (2, 1)
        """
        # print(shape)
        # index = index[:, :shape]
        # x = x[:, :shape, :]
        return self.detector(index, self.wave(x))
