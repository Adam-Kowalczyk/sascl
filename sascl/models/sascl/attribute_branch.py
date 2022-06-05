from torch import nn

from sascl.models.common.common import FeedForwardResidual, MLP
from sascl.models.common.scattering_transform import ScatteringTransform


class AttributeBranch(nn.Module):
    def __init__(self,
               set_size,
               conv_output_dim,
               attr_heads,
               attr_net_hidden_dims,
               attr_output_dim,
               rel_heads,
               rel_net_hidden_dims,
               total_rules):
        super().__init__()

        self.attr_heads = attr_heads
        self.attr_net = ScatteringTransform([conv_output_dim, *attr_net_hidden_dims, attr_output_dim], heads = attr_heads)
        self.ff_residual = FeedForwardResidual(attr_output_dim)

        self.rel_heads = rel_heads
        self.rel_net = MLP(set_size * (attr_output_dim // rel_heads), *rel_net_hidden_dims)

        self.to_logit = nn.Linear(rel_net_hidden_dims[-1] * rel_heads, total_rules + 1)

    def forward(self, features, b, m, n):

        attrs = self.attr_net(features)
        attrs = self.ff_residual(attrs)

        attrs = attrs.reshape(b, m, n, self.rel_heads, -1).transpose(-2, -3).flatten(3)
        rels = self.rel_net(attrs)
        rels = rels.flatten(2)

        logits = self.to_logit(rels).flatten(1)
        return logits