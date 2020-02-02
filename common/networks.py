"""class for network(Temporal Convolutional Network)"""

import torch.nn as nn


class FirstTemporalBlock(nn.Module):
    def __init__(self, n_input, channels, kernel_size, stride, dilation,
                 padding, dropout):
        super(FirstTemporalBlock, self).__init__()
        self.conv = nn.Conv1d(n_input, channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.bn, self.relu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out


class TemporalBlock(nn.Module):
    """Block of mini temporal convolutional networks.
    [1dconv->batch_norm->relu->dropout] x2
    """

    def __init__(self, channels, kernel_size, stride, dilation, padding,
                 pad, dropout, one_frame):
        super(TemporalBlock, self).__init__()
        self.one_frame = one_frame
        self.kernel_size = kernel_size
        self.pad = pad
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, 1, stride=1,
                               padding=padding, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels, momentum=0.1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.relu2, self.dropout2)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # slice the residuals(left and right, symmetrically) to match the shape of subsequent tensors.
        if self.one_frame:
            res = x[..., self.kernel_size // 2:: self.kernel_size]
        else:
            res = x[..., self.pad: x.shape[2] - self.pad]
        out = self.net(x)
        return out + res


class TemporalConvNet(nn.Module):
    """Predictor for 3d keypoints.
    """

    def __init__(self, num_joint_2d, num_joint_3d, in_features, kernel_size, stride, init_dilation, padding,
                 dropout, num_block, channels, train):
        """Initialize this model.

        Args:
            num_joint_2d (int):
            in_features (int):
            kernel_size (int):
            stride (int):
            init_dilation (int):
            padding (int):
            dropout (float):
            num_block (int):
            channels (int):
            train (bool):
        """
        super(TemporalConvNet, self).__init__()
        self.num_joint_3d = num_joint_3d
        # First block(no residual structure)
        if train:
            layers = [FirstTemporalBlock(n_input=num_joint_2d * in_features, channels=channels,
                                         kernel_size=kernel_size, stride=kernel_size,
                                         dilation=init_dilation, padding=padding, dropout=dropout)]
        else:
            layers = [FirstTemporalBlock(n_input=num_joint_2d * in_features, channels=channels,
                                         kernel_size=kernel_size, stride=stride,
                                         dilation=init_dilation, padding=padding, dropout=dropout)]
        dilation = init_dilation
        for i in range(num_block):
            dilation *= kernel_size
            pad = ((kernel_size - 1) * dilation) // 2
            if train:
                layers += [TemporalBlock(channels=channels, kernel_size=kernel_size, stride=kernel_size,
                                         dilation=1, padding=padding, pad=pad, dropout=dropout,
                                         one_frame=True)]
            else:
                layers += [TemporalBlock(channels=channels, kernel_size=kernel_size, stride=stride,
                                         dilation=dilation, padding=padding, pad=pad, dropout=dropout,
                                         one_frame=False)]
        # shrink
        layers.append(nn.Conv1d(channels, num_joint_3d * 3, 1))

        self.network = nn.Sequential(*layers)
        self.layers = layers

    def set_batch_norm_momentum(self, momentum):
        self.layers[0].bn.momentum = momentum
        for i in range(1, len(self.layers) - 1):
            self.layers[i].bn1.momentum = 0.01
            self.layers[i].bn2.momentum = 0.01

    def forward(self, x):
        # (*, T, num_joint_2d, 2)->(*, num_joint_2d * 2, T)
        batch_size = x.shape[0]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # TCN process
        x = self.network(x)

        # (*, num_joint_3d * 3, T)->(*, T, num_joint_3d * 3)
        x = x.permute(0, 2, 1)
        out = x.view(batch_size, -1, self.num_joint_3d, 3)

        return out
