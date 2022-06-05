from torch import nn

from sascl.models.common.common import MLP

class ScatteringTransform(nn.Module):
    def __init__(self, dims, heads, activation = None):
        super().__init__()
        assert len(dims) > 2, 'must have at least 3 dimensions, for dimension in, the hidden dimension, and dimension out'

        dim_in, *hidden_sizes, dim_out = dims

        dim_in //= heads
        dim_out //= heads

        self.heads = heads
        self.mlp = MLP(dim_in, *hidden_sizes, dim_out, activation = activation)

    def forward(self, x):
        shape, heads = x.shape, self.heads
        dim = shape[-1]

        assert (dim % heads) == 0, f'the dimension {dim} must be divisible by the number of heads {heads}'

        x = x.reshape(-1, heads, dim // heads)
        x = self.mlp(x)

        return x.reshape(*shape[:-1], -1)