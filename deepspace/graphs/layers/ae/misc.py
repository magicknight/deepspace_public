from torch import nn


class ParallelBatchNorm1d(nn.BatchNorm1d):

    def forward(self, x):
        x_reshaped = x.view(-1, x.shape[2])
        y_reshaped = super().forward(x_reshaped)
        y = y_reshaped.view(*x.shape)
        return y

    @staticmethod
    def _get_name():
        return 'ParallelBatchNorm1d'


def fc_layer(in_features, out_features, activation=None, batchnorm=True):
    layers = [nn.Linear(in_features, out_features)]

    if activation is not None:
        layers += [activation]
    if batchnorm:
        layers += [ParallelBatchNorm1d(out_features)]

    return nn.Sequential(*layers)
