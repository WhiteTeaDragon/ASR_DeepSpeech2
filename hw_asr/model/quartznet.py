import torch
from torch import nn

from hw_asr.base import BaseModel


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding="same", bias=False):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels,
                                   kernel_size=kernel_size, groups=in_channels,
                                   bias=bias, padding=padding, stride=stride)
        self.pointwise = nn.Conv1d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class QuartzModule(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, relu=True, stride=1):
        super(QuartzModule, self).__init__()
        self.conv_layer = SeparableConv1d(c_in, c_out, kernel_size,
                                          stride=stride, bias=False)
        self.normalization = nn.BatchNorm1d(c_out)
        self.relu = relu
        if self.relu:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.normalization(x)
        if self.relu:
            x = self.activation(x)
        return x


class QuartzBlock(nn.Module):
    def __init__(self, num_of_repeats_inside_block, c_in, c_out, kernel_size):
        super(QuartzBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_out)
        )
        base_modules = []
        for i in range(num_of_repeats_inside_block):
            not_last_one = (i + 1 != num_of_repeats_inside_block)
            base_modules.append(QuartzModule(c_in, c_out, kernel_size,
                                             relu=not_last_one))
            if c_in != c_out:
                c_in = c_out
        self.base_modules = nn.Sequential(*base_modules)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        after_module = self.base_modules(x)
        x = residual + after_module
        x = self.final_relu(x)
        return x


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class, num_of_block_repeats,
                 num_of_repeats_inside_blocks, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.c1 = QuartzModule(n_feats, 256, 33, relu=True)
        channels = [256, 256, 256, 512, 512, 512]
        kernel_sizes = [33, 39, 51, 63, 75]
        blocks = []
        for i in range(5):
            for j in range(num_of_block_repeats):
                # in_index == i if j == 0, i + 1 otherwise
                in_index = i + (j + num_of_block_repeats - 1) // \
                           num_of_block_repeats
                blocks.append(QuartzBlock(num_of_repeats_inside_blocks,
                                          channels[in_index], channels[i + 1],
                                          kernel_sizes[i]))
        self.blocks = nn.Sequential(*blocks)
        self.c2 = QuartzModule(512, 512, 87, relu=True)
        self.c3 = QuartzModule(512, 1024, 1, relu=True)
        self.c4 = nn.Conv1d(1024, n_class, 1, dilation=2)

    def forward(self, spectrogram, *args, **kwargs):
        x = self.c1(torch.transpose(spectrogram, 1, 2))
        x = self.blocks(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return {"logits": torch.transpose(x, 1, 2)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
