import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.runner import BaseModule, _load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES
import math

class New_GAU(nn.Module):
    def __init__(self, dim = 96, expansion_factor = 2., qkv_bias=False,
                sr_ratio=[8, 4, 2, 1], num_heads=2):
        super().__init__()

        hidden_dim = int(expansion_factor * dim)
        query_key_dim = dim // 2
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim)
        )

        self.proj = nn.Linear(dim, dim)
        self.pos = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sig = self.sigmoid = nn.Sigmoid()

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio==8:
                self.sr1_num = sr_ratio
                self.sr2_num = sr_ratio // 2
                self.ratio1 = (self.sr1_num) ** 2
                self.ratio2 = (self.sr2_num) ** 2
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=self.sr1_num, stride=self.sr1_num)
                self.norm1 = nn.LayerNorm(dim)
                self.gamma1 = nn.Parameter(torch.ones(2, query_key_dim))
                self.beta1 = nn.Parameter(torch.zeros(2, query_key_dim))
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=self.sr2_num, stride=self.sr2_num)
                self.norm2 = nn.LayerNorm(dim)
                self.gamma2 = nn.Parameter(torch.ones(2, query_key_dim))
                self.beta2 = nn.Parameter(torch.zeros(2, query_key_dim))
            if sr_ratio==4:
                self.sr1_num = sr_ratio
                self.sr2_num = sr_ratio // 2
                self.ratio1 = (self.sr1_num) ** 2
                self.ratio2 = (self.sr2_num) ** 2
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=self.sr1_num, stride=self.sr1_num)
                self.norm1 = nn.LayerNorm(dim)
                self.gamma1 = nn.Parameter(torch.ones(2, query_key_dim))
                self.beta1 = nn.Parameter(torch.zeros(2, query_key_dim))
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=self.sr2_num, stride=self.sr2_num)
                self.norm2 = nn.LayerNorm(dim)
                self.gamma2 = nn.Parameter(torch.ones(2, query_key_dim))
                self.beta2 = nn.Parameter(torch.zeros(2, query_key_dim))
            if sr_ratio==2:
                self.sr1_num = sr_ratio
                self.sr2_num = sr_ratio // 2
                self.ratio1 = (self.sr1_num) ** 2
                self.ratio2 = (self.sr2_num) ** 2
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=self.sr1_num, stride=self.sr1_num)
                self.norm1 = nn.LayerNorm(dim)
                self.gamma1 = nn.Parameter(torch.ones(2, query_key_dim))
                self.beta1 = nn.Parameter(torch.zeros(2, query_key_dim))
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=self.sr2_num, stride=self.sr2_num)
                self.norm2 = nn.LayerNorm(dim)
                self.gamma2 = nn.Parameter(torch.ones(2, query_key_dim))
                self.beta2 = nn.Parameter(torch.zeros(2, query_key_dim))
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.kv = nn.Linear(dim, dim, bias=qkv_bias)
            self.gamma = nn.Parameter(torch.ones(2, dim))
            self.beta = nn.Parameter(torch.zeros(2, dim))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)

        if self.sr_ratio > 1:
                x_ = normed_x.permute(0, 2, 1).reshape(B, C, H, W)
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))

                Z1 = self.to_qk(x_1)
                QK1 = einsum('... d, h d -> ... h d', Z1, self.gamma1) + self.beta1
                q1, k1 = QK1.unbind(dim=-2)

                Z2 = self.to_qk(x_2)
                QK2 = einsum('... d, h d -> ... h d', Z2, self.gamma2) + self.beta2
                q2, k2 = QK2.unbind(dim=-2)
                attn1 = (v[:, :, :C // self.num_heads] @ q1.transpose(-2, -1)) / N
                A1 = F.relu(attn1) ** 2
                V1 = einsum('b i j, b j d -> b i d', A1, k1)

                attn2 = (v[:, :, C // self.num_heads:] @ q2.transpose(-2, -1)) / N
                A2 = F.relu(attn2) ** 2
                V2 = einsum('b i j, b j d -> b i d', A2, k2)

                x = torch.cat([V1, V2], dim=-1)
                value = x * gate

                out = self.proj(value)

                output = out + x

        else:
            Z = self.kv(normed_x)
            QK = einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
            q, k = QK.unbind(dim=-2)

            attn = (v @ q.transpose(-2, -1)) / N
            A = F.relu(attn) ** 2
            V = einsum('b i j, b j d -> b i d', A, k)
            value = V * gate
            out = self.proj(value)
            output = out + x

        return output

class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=[8, 4, 2, 1], drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.new_attn = New_GAU(dim=dim, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.new_attn(self.norm1(x), H, W))
        #x = self.norm1(x + self.new_attn(x, H, W))
        #x = self.drop_path(self.norm2(x))
        return x

class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True) #depth-wise conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class ConvStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=4, with_pos=True):
        super().__init__()

        self.patch_size = to_2tuple(patch_size)
        stem = []
        in_dim, out_dim = in_ch, out_ch

        self.proj1 = nn.Sequential(
                                   nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                   #nn.BatchNorm2d(out_dim),
                                   nn.GroupNorm(num_groups=32, num_channels=out_dim),
                                   #nn.ReLU(inplace=True))
                                   nn.GELU())

        stem.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False))
        stem.append(nn.GroupNorm(num_groups=32, num_channels=out_dim))
        #stem.append(nn.BatchNorm2d(out_dim))
        #stem.append(nn.ReLU(inplace=True))
        stem.append(nn.GELU())

        #stem.append(nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1))
        self.proj = nn.Sequential(*stem)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_dim)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj1(x)
        x1 = x

        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W), x1

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, out_ch=96, patch_size=4, with_pos=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size // 2, groups=32)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=in_ch)
        self.linear = nn.Linear(out_ch, out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm2(x)
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        x = self.linear(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #    ﶨ   ˲в          2
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True))

                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size
    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)
        return x

class unetUp1(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp1, self).__init__()
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(out_size, out_size, 1)

    def forward(self, high_feature, low_feature, class_feature):
        high_feature = self.up(high_feature)
        outputs = torch.cat([high_feature, low_feature, class_feature], 1)
        final = self.pixelshuffle(outputs)
        return self.conv(final)


class unetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp2, self).__init__()
        self.select_class = nn.Sequential(
            nn.Conv2d(in_size, out_size, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        self.conv = nn.Conv2d(out_size, out_size, 1)

    def forward(self, high_feature, low_feature, class_feature):
        high_feature = self.select_class(high_feature)
        outputs = torch.cat([high_feature, low_feature, class_feature], 1)
        final = self.pixelshuffle(outputs)
        return self.conv(final)

class unetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp3, self).__init__()
        self.conv = unetConv2(in_size * 3, in_size, True, n=1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, high_feature, low_feature, class_feature):
        outputs = torch.cat([high_feature, low_feature, class_feature], 1)
        #outputs = torch.cat([high_feature, low_feature], 1)
        #final = self.pixelshuffle(outputs)
        return self.conv(outputs)


@BACKBONES.register_module()
class HybridFormer(BaseModule):
    def __init__(self, in_chans=3, num_classes=5, num_heads=[1, 2, 4, 8],
                 embed_dims=[32, 64, 128, 256],  drop_path_rate=0.,
                 depths=[1, 1, 1, 1], sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3),
                 pretrained=None, init_cfg=None):
        super().__init__()
        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        self.depths = depths
        self.out_indices = out_indices

        self.stem = ConvStem(in_chans, embed_dims[0], patch_size=4)
        self.patch_2 = PatchEmbed(embed_dims[0], embed_dims[1], patch_size=2)
        self.patch_3 = PatchEmbed(embed_dims[1], embed_dims[2], patch_size=2)
        self.patch_4 = PatchEmbed(embed_dims[2], embed_dims[3], patch_size=2)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], sr_ratios[0], dpr[cur + i])
            for i in range(depths[0])
        ])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], sr_ratios[1], dpr[cur + i])
            for i in range(depths[1])
        ])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + i])
            for i in range(depths[2])
        ])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], sr_ratios[3], dpr[cur + i])
            for i in range(depths[3])
        ])

        for idx in out_indices:
            out_ch = embed_dims[idx]
            layer = LayerNorm(out_ch, eps=1e-6, data_format="channels_first")
            layer_name = f"norm_{idx + 1}"
            self.add_module(layer_name, layer)

        filters = embed_dims

        self.up_concat4 = unetUp1(filters[3], filters[2])
        self.up_concat3 = unetUp2(filters[2], filters[1])
        self.up_concat2 = unetUp2(filters[1], filters[0])
        self.up_concat1 = unetUp3(filters[0], num_classes)

        self.final = nn.Conv2d(filters[0], num_classes, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(in_channels=filters[0], out_channels=filters[0],
                                         kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes,
                                    kernel_size=1, stride=1)

        #------------------------------------------------------------------------#
        #------------------------------  resnet-------------------------------#

        self.conv1 = nn.Sequential(nn.Conv2d(in_chans, filters[0], 3, 1, 1),
                                   nn.BatchNorm2d(filters[0]),
                                   nn.ReLU(inplace=True),

                                   nn.Conv2d(filters[0], filters[0], 3, 1, 1),
                                   nn.BatchNorm2d(filters[0]),
                                   nn.ReLU(inplace=True)
                                   )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        blocks = [2, 2, 2, 2]
        self.layer1 = self.make_layer(ResBlock, filters[0], filters[0], blocks[0], stride=1)
        self.layer2 = self.make_layer(ResBlock, filters[0], filters[1], blocks[1], stride=2)
        self.layer3 = self.make_layer(ResBlock, filters[1], filters[2], blocks[2], stride=2)
        self.layer4 = self.make_layer(ResBlock, filters[2], filters[3], blocks[3], stride=2)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # encoder path
        inputs = x

        #----------------------------------------------#
        #------------------ GAU -----------------------#

        outs = []
        B, _, H, W = x.shape
        x, (H, W), x1 = self.stem(x)
        outs.append(self.norm_1(x1))   #(B, 32, 128, 256)

        #embeddings = []
        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
            #embeddings.append(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        if 0 in self.out_indices:
            outs.append(self.norm_1(x))   # (B, 64, 64, 128)

        # stage 2
        x, (H, W) = self.patch_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
            #embeddings.append(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        if 1 in self.out_indices:
            outs.append(self.norm_2(x))   # (B, 128, 32, 64)

        # stage 3
        x, (H, W) = self.patch_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
            #embeddings.append(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        if 2 in self.out_indices:
            outs.append(self.norm_3(x))   # (B, 256, 16, 32)

        # stage 4
        x, (H, W) = self.patch_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
            #embeddings.append(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        if 3 in self.out_indices:
            outs.append(self.norm_4(x))  # (B, 512, 8, 16)

        # ----------------------------------------------#
        # ----------------- resnet18  ----------------#
        conv1 = self.conv1(inputs)
        max1 = self.maxpool(conv1)
        conv2 = self.layer1(max1)
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        center = self.layer4(conv4)

        # decoder path
        up4 = self.up_concat4(outs[4], outs[3], center)
        up3 = self.up_concat3(up4, outs[2], conv4)
        up2 = self.up_concat2(up3, outs[1], conv3)
        up1 = self.up_concat1(up2, outs[0], conv2)
        up1 = self.deconv(up1)
        final = self.final(up1)

        #final = self.deconv(final)
        #final = self.final_conv(final)
        return final

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, dim, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.dim = (dim,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.dim, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True) # з     ƽ
            s = (x - u).pow(2).mean(1, keepdim=True) #
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


if __name__ == "__main__":

    model = HybridFormer()
    dummy_input = torch.randn(1, 3, 256, 512)
    output = model(dummy_input)
    print(output.size())

    from thop import profile
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"Total Params: {params}")
    print(f"Total FLOPs: {flops / 1e9} GFLOPs")