import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class CoordXY(Module):
    """
    This operation next to convolution provides CoordConv layer result from paper https://arxiv.org/pdf/1807.03247.pdf.
    
    Parameters:
    -----------
    out_channels : int
        number of output channels
    std : float
        standard deviation for initialization of parameters
    """
    def __init__(self, out_channels : int, std : float=0.1):
        super(CoordXY, self).__init__()
        self.out_channels = out_channels
        self.std = std
        self.alpha = Parameter((torch.randn(out_channels) * self.std).view(1, self.out_channels, 1, 1))
        self.beta = Parameter((torch.randn(out_channels) * self.std).view(1, self.out_channels, 1, 1))

    def forward(self, input):
        assert len(input.shape) == 4, "Tensor should have 4 dimensions"
        dimx = input.shape[-1]
        dimy = input.shape[-2]
        x_special = torch.linspace(-1, 1, steps=dimx).view(1, 1, 1, dimx)
        y_special = torch.linspace(-1, 1, steps=dimy).view(1, 1, dimy, 1)
        if self.alpha.is_cuda:
            device = self.alpha.get_device()
            x_special = x_special.cuda(device=device)
            y_special = y_special.cuda(device=device)

        input.add_(self.alpha * x_special).add_(self.beta * y_special)
        return input
