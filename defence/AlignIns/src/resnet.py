import torch.nn as nn


def conv_block(in_channels, out_channels, pool=False, track_running_stats=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats),
        #   nn.GroupNorm(out_channels // 2, out_channels),
        nn.ReLU(inplace=True),
    ]

    #   nn.BatchNorm2d(out_channels,track_running_stats=track_running_statsResNet9,
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, args):
        super().__init__()

        self.conv1 = conv_block(
            in_channels,
            64,
        )
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes, bias=False)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
