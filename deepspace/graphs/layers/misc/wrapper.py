from torch import nn

# dummy network wrapper. for adding multiple networks into tensorboardX


class Wrapper(nn.Module):
    def __init__(
        self,
        network_1,
        network_2,
    ):
        super().__init__()
        self.network_1 = network_1
        self.network_2 = network_2

    def forward(self, data):
        # Perform a forward pass through all the networks and return the result
        if isinstance(data, tuple):
            return self.network_1(data[0]), self.network_2(data[1])
        else:
            return self.network_1(data), self.network_2(data)
