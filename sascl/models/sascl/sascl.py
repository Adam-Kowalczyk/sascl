import torch
from torch import nn

from sascl.models.common.common import ConvNet
from sascl.models.sascl.attribute_branch import AttributeBranch

class SASCL(nn.Module):
    def __init__(
        self,
        image_size = 160,
        set_size = 9,
        conv_channels = [1, 16, 16, 32, 32, 32],
        conv_output_dim = 80,
        attr_heads = 10,
        attr_net_hidden_dims = [64],
        attr_output_dim = 40,
        rel_heads = 40,
        rel_net_hidden_dims = [23, 5],
        attr_branches = 10,
        total_rules = 4):

        super().__init__()
        self.vision = ConvNet(image_size, conv_channels, conv_output_dim)
        self.attr_branches = attr_branches

        for i in range(attr_branches):
            branch = AttributeBranch(set_size, conv_output_dim, attr_heads,
                                   attr_net_hidden_dims, attr_output_dim, rel_heads, 
                                   rel_net_hidden_dims, total_rules)
            setattr(self, "attr_branch%d" % i, branch)

    def forward(self, sets):
        b, m, n, c, h, w = sets.shape
        images = sets.view(-1, c, h, w)
        features = self.vision(images)

        all_logits=[]
        for i in range(self.attr_branches):
            all_logits.append(getattr(self, "attr_branch%d" % i)(features, b, m, n))

        return torch.stack(all_logits, dim=2)