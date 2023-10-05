import torch
import torch.nn as nn


class Unet2D_encoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_filter):
        '''This is the UNet that is used in the original version of Privacy-Net. Slightly adapted from 3D to 2D.
        Taken from https://github.com/bachkimn/Privacy-Net-An-Adversarial-Approach-forIdentity-Obfuscated-Segmentation-of-MedicalImages.

        :param in_dim: int
            Specifies the number of input channels.
        :param out_dim: int
            Specifies the number of output channels.
        :param num_filters: int
            Specifies the number of filters in the first conv block.
        '''

        super(Unet2D_encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        print("|------Initiating Encryptor: 2D-U-Net ------|")

        self.down_1 = conv_block_2_2d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_2d()
        self.down_2 = conv_block_2_2d(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool_2d()
        self.down_3 = conv_block_2_2d(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = maxpool_2d()

        self.bridge = conv_block_2_2d(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = conv_trans_block_2d(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = conv_block_2_2d(self.num_filter * 12, self.num_filter * 4, act_fn)
        self.trans_2 = conv_trans_block_2d(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = conv_block_2_2d(self.num_filter * 6, self.num_filter * 2, act_fn)
        self.trans_3 = conv_trans_block_2d(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = conv_block_2_2d(self.num_filter * 3, self.num_filter * 1, act_fn)

        self.out = conv_block_2d(self.num_filter, out_dim, nn.Sigmoid())

    def forward(self, x):
        # print('x:{}'.format(x.shape))
        down_1 = self.down_1(x)
        # print('down_1:{}'.format(down_1.shape))
        pool_1 = self.pool_1(down_1)
        # print('pool_1:{}'.format(pool_1.shape))
        down_2 = self.down_2(pool_1)
        # print('down_2:{}'.format(down_2.shape))
        pool_2 = self.pool_2(down_2)
        # print('pool_2:{}'.format(pool_2.shape))
        down_3 = self.down_3(pool_2)
        # print('down_2:{}'.format(down_3.shape))
        pool_3 = self.pool_3(down_3)
        # print('pool_3:{}'.format(pool_3.shape))

        bridge = self.bridge(pool_3)
        # print('bridge:{}'.format(bridge.shape))

        trans_1 = self.trans_1(bridge)
        # print('trans_1:{}'.format(trans_1.shape))
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        # print('concat_1:{}'.format(concat_1.shape))
        up_1 = self.up_1(concat_1)
        # print('up_1:{}'.format(up_1.shape))
        trans_2 = self.trans_2(up_1)
        # print('trans_2:{}'.format(trans_2.shape))
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        # print('concat_2:{}'.format(concat_2.shape))
        up_2 = self.up_2(concat_2)
        # print('up_2:{}'.format(up_2.shape))
        trans_3 = self.trans_3(up_2)
        # print('trans_3:{}'.format(trans_3.shape))
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        # print('concat_3:{}'.format(concat_3.shape))
        up_3 = self.up_3(concat_3)
        # print('up_3:{}'.format(up_3.shape))

        out = self.out(up_3)
        # print('out:{}'.format(out.shape))
        return out


def conv_block_2d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block_2d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool_2d():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2_2d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block_2d(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


def conv_block_3_2d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block_2d(in_dim, out_dim, act_fn),
        conv_block_2d(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model
