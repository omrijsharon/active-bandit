# from .types_ import *
from torch import nn
from abc import abstractmethod
from torch import Tensor


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor):
        raise NotImplementedError

    def decode(self, input: Tensor) -> any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs):
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: any, **kwargs) -> Tensor:
        pass