from torch import nn

def default(val, default_val):
    return val if val is not None else default_val

class MLP(nn.Module):
    def __init__(self, *dims, activation = None):
        super().__init__()
        assert len(dims) > 2, 'must have at least 3 dimensions, for dimension in and dimension out'
        activation = default(activation, nn.ReLU)

        layers = []
        pairs = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(pairs):
            is_last = ind >= (len(pairs) - 1)
            layers.append(nn.Linear(dim_in, dim_out))
            if not is_last:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FeedForwardResidual(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.LayerNorm(dim * mult),
            nn.ReLU(inplace = True),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return x + self.net(x)

class ConvNet(nn.Module):
    def __init__(self, image_size, chans, output_dim):
        super().__init__()

        num_conv_layers = len(chans) - 1
        conv_output_size = image_size // (2 ** num_conv_layers)

        convolutions = []
        channel_pairs = list(zip(chans[:-1], chans[1:]))

        for ind, (chan_in, chan_out) in enumerate(channel_pairs):
            is_last = ind >= (len(channel_pairs) - 1)
            convolutions.append(nn.Conv2d(chan_in, chan_out, 3, padding=1, stride=2))
            if not is_last:
                convolutions.append(nn.BatchNorm2d(chan_out))

        self.net = nn.Sequential(
            *convolutions,
            nn.Flatten(1),
            nn.Linear(chans[-1] * (conv_output_size ** 2), output_dim),
            nn.ReLU(inplace=True),
            FeedForwardResidual(output_dim)
        )

    def forward(self, x):
        return self.net(x)