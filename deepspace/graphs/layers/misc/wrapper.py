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
        q1 = self.network_1(data)
        q2 = self.network_2(data)
        return q1, q2
