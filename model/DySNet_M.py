'''
ModeT

Original code retrieved from:
https://github.com/ZAX130/SmileCode

Original paper:
Wang, Haiqiao, Dong Ni, and Yi Wang. "ModeT: Learning deformable image registration via motion decomposition transformer." International conference on medical image computing and computer-assisted intervention. Cham: Springer Nature Switzerland, 2023.

Modified and tested by:
Shaochen Bi
bisc0507@163.com
HKUST
'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F

class Mlp(nn.Module):

    def __init__(self, in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialTransformer(nn.Module):


    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid 2D
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]  # 2D shape

        # normalize to [-1, 1]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class DySA(nn.Module):
    """
    AdSB is included in DySA.
    This is the single-point version of DySNet. The 3×3 version of DySNet can be found in DySNet_X.
    """

    def __init__(self, input_dim, amp, num_heads, qkv_bias=False, q_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.amp = amp
        self.input_dim = input_dim
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = q_scale or head_dim ** -0.5

        self.kv = nn.Conv2d(input_dim, input_dim * 2, kernel_size=1, bias=qkv_bias)
        self.q = nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=1)

        self.offset_predictor = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, 2, kernel_size=1, bias=True),  
            )
        
    def forward_cross(self, x_kv, x_q):
        """
        AdSB is included in DySA.
        """
        B, C, H, W = x_q.shape
        q = self.q(x_q).reshape(B, C // self.num_heads, self.num_heads, H, W)
        q = q * self.scale

        #This is the beginning of AdSB!!!
        offset_input = torch.cat([x_q, x_kv], dim=1)  # (B, 2*C, H, W)

        offset = self.offset_predictor(offset_input)  # (B, 2, H, W)
        offset = offset.reshape(B, 1, 2, H, W)  # (B, 1, 2, H, W)

        grid = offset.permute(0, 1, 3, 4, 2).contiguous()  # (B, 1, H, W, 2)

        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1 

        x_kv_reshape = x_kv.reshape(B * 1, C, H, W)  # (B, C, H, W)
        grid_reshape = grid.reshape(B * 1, H, W, 2)  # (B, H, W, 2)

        x_kv_sampled = F.grid_sample(x_kv_reshape, grid_reshape, mode='bilinear', padding_mode='zeros', align_corners=True)

        x_kv_sampled = x_kv_sampled.view(B, 1, C, H, W)

        x_kv_sampled_reshape = x_kv_sampled.view(B * 1, C, H, W)  # (B, C, H, W)
        kv = self.kv(x_kv_sampled_reshape)  # (B, 2*C, H, W)

        kv = kv.reshape(B, 1, 2, C // self.num_heads, self.num_heads, H, W)

        k = kv[:, 0, 0]  # (B, C_head, num_heads, H, W)
        v = kv[:, 0, 1]  # (B, C_head, num_heads, H, W)

        pad_dim = (self.amp, self.amp, self.amp, self.amp)
        k_padded = F.pad(k, pad_dim, "constant")
        v_padded = F.pad(v, pad_dim, "constant")
        #This is the ending of AdSB!!!

        B, C_head, num_heads, H, W = q.shape

        center = self.amp
        i, j = center, center 

        k_patch = k_padded[:, :, :, i:i + H, j:j + W]  # (B, C_head, num_heads, H, W)

        attn = torch.sum(q * k_patch, dim=1, keepdim=True)  # (B, 1, num_heads, H, W)

        attn = self.softmax(attn)    # (B, 1, num_heads, H, W)
        attn = self.attn_drop(attn)

        v_patch = v_padded[:, :, :, i:i + H, j:j + W]  # (B, C_head, num_heads, H, W)
        v_ = attn * v_patch  # (B, C_head, num_heads, H, W)

        v_ = v_.permute(0, 1, 2, 3, 4).contiguous()  # (B, C_head, num_heads, H, W)
        v_ = v_.reshape(B, C_head * num_heads, H, W)

        x_v = self.proj(v_)
        x_v = self.proj_drop(x_v)

        return x_v

    def forward(self, x_q, x_kv=None, mode='cross'):
        return self.forward_cross(x_q, x_kv)


class DSB(nn.Module):
    def __init__(self,
                 dim,
                 amp,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.GroupNorm,
                 use_checkpoint=False
                 ):
        super().__init__()
        self.dim = dim
        self.amp = amp
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(1, dim)
        self.attn = DySA(dim, amp=amp, num_heads=num_heads, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=drop)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(1, dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward_atten(self, x_q, x_kv, mode='cross'):
        x_q = self.norm1(x_q)
        if mode == 'cross':
            x_kv = self.norm1(x_kv)
        x_v = self.attn(x_q, x_kv, mode=mode)
        x_v = x_v.contiguous()
        return x_v

    def forward_mlp(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x_q, x_kv, mode='cross'):
        shortcut = x_q

        if self.use_checkpoint:
            x_v = checkpoint.checkpoint(self.forward_atten, x_q, x_kv, mode)
        else:
            x_v = self.forward_atten(x_q, x_kv, mode)
        x_q = shortcut + self.drop_path1(x_v)

        shortcut = x_q
        if self.use_checkpoint:
            x_q = checkpoint.checkpoint(self.forward_mlp, x_q)
        else:
            x_q = self.forward_mlp(x_q)
        x_q = shortcut + self.drop_path2(x_q)
        return x_q
    

class VecInt(nn.Module):

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ConvInsBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.actout = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class DeconvBlock(nn.Module):
    def __init__(self, dec_channels, skip_channels):
        super(DeconvBlock, self).__init__()
        self.upconv = UpConvBlock(dec_channels, skip_channels)
        self.conv = nn.Sequential(
            ConvInsBlock(2 * skip_channels, skip_channels),
            ConvInsBlock(skip_channels, skip_channels)
        )

    def forward(self, dec, skip):
        dec = self.upconv(dec)
        out = self.conv(torch.cat([dec, skip], dim=1))
        return out


class Encoder(nn.Module):

    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvBlock(in_channel, c),
            ConvInsBlock(c, 2 * c),
            ConvInsBlock(2 * c, 2 * c)
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvInsBlock(8 * c, 16 * c),
            ConvInsBlock(16 * c, 16 * c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvInsBlock(16 * c, 32 * c),
            ConvInsBlock(32 * c, 32 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        return out0, out1, out2, out3, out4


class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        # 2D tensor: (B, C, H, W) -> (B, H, W, C)
        feat = feat.permute(0, 2, 3, 1)
        feat = self.norm(self.proj(feat))
        return feat


class CWM(nn.Module):
    def __init__(self, in_channels, channels):
        super(CWM, self).__init__()

        c = channels
        self.num_fields = in_channels // 2  

        self.conv = nn.Sequential(
            ConvInsBlock(in_channels, channels, 3, 1),
            ConvInsBlock(channels, channels, 3, 1),
            nn.Conv2d(channels, self.num_fields, 3, 1, 1),
            nn.Softmax(dim=1)
        )

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )

    def forward(self, x):

        x = self.upsample(x)
        weight = self.conv(x)

        weighted_field = 0

        for i in range(self.num_fields):
            w = x[:, 2 * i: 2 * (i + 1)]
            weight_map = weight[:, i:(i + 1)]
            weighted_field = weighted_field + w * weight_map

        return 2 * weighted_field

class GWM(nn.Module):
    def __init__(self, in_channels, channels):
        super(GWM, self).__init__()

        self.num_fields = in_channels // 2

        self.feature_extractor = nn.Sequential(
            ConvInsBlock(in_channels, channels, 3, 1),
            ConvInsBlock(channels, channels, 3, 1),
        )

        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels, self.num_fields, 3, 1, 1),
            nn.Softmax(dim=1)
        )

        self.gate_conv = nn.Conv2d(channels, self.num_fields, 3, 1, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        features = self.feature_extractor(x)

        weight = self.weight_conv(features) 
        gate = torch.sigmoid(self.gate_conv(features))  

        weighted_field = 0
        for i in range(self.num_fields):
            w = x[:, 2 * i: 2 * (i + 1)] 
            gated_weight = weight[:, i:(i + 1)] * gate[:, i:(i + 1)]  
            weighted_field = weighted_field + w * gated_weight

        return 2 * weighted_field

class GWM_noup(nn.Module):
    def __init__(self, in_channels, channels):
        super(GWM_noup, self).__init__()

        self.num_fields = in_channels // 2

        self.feature_extractor = nn.Sequential(
            ConvInsBlock(in_channels, channels, 3, 1),
            ConvInsBlock(channels, channels, 3, 1),
        )

        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels, self.num_fields, 3, 1, 1),
            nn.Softmax(dim=1)
        )

        self.gate_conv = nn.Conv2d(channels, self.num_fields, 3, 1, 1)


    def forward(self, x):
        features = self.feature_extractor(x)

        weight = self.weight_conv(features)   
        gate = torch.sigmoid(self.gate_conv(features))  

        weighted_field = 0
        for i in range(self.num_fields):
            w = x[:, 2 * i: 2 * (i + 1)] 
            gated_weight = weight[:, i:(i + 1)] * gate[:, i:(i + 1)]
            weighted_field = weighted_field + w * gated_weight

        return 2 * weighted_field


class CWM_noup(nn.Module):
    def __init__(self, in_channels, channels):
        super(CWM_noup, self).__init__()

        c = channels
        self.num_fields = in_channels // 2

        self.conv = nn.Sequential(
            ConvInsBlock(in_channels, channels, 3, 1),
            ConvInsBlock(channels, channels, 3, 1),
            nn.Conv2d(channels, self.num_fields, 3, 1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        weight = self.conv(x)

        weighted_field = 0

        for i in range(self.num_fields):
            w = x[:, 2*i: 2*(i+1)]
            weight_map = weight[:, i:(i+1)]
            weighted_field = weighted_field + w*weight_map

        return 2*weighted_field


class DySNet_M(nn.Module):
    def __init__(self,
                 inshape=(160, 160),
                 in_channel=1,
                 channels=4,
                 head_dim=6,
                 amp=0,
                 num_heads=[8, 4, 2, 1, 1],
                 scale=None):
        super(DySNet_M, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.amp = amp
        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.projblock1 = ProjectionLayer(2*c, dim=head_dim*num_heads[4])

        self.projblock2 = ProjectionLayer(4*c, dim=head_dim*num_heads[3])
        self.cwm2 = GWM_noup(6 * num_heads[3], 3 * num_heads[3] * 2)

        self.projblock3 = ProjectionLayer(8*c, dim=head_dim*num_heads[2])
        self.cwm3 = GWM(6 * num_heads[2], 3 * num_heads[2] * 2)

        self.projblock4 = ProjectionLayer(16*c, dim=head_dim*num_heads[1])
        self.cwm4 = GWM(6 * num_heads[1], 3 * num_heads[1] * 2)

        self.projblock5 = ProjectionLayer(32*c, dim=head_dim*num_heads[0])
        self.cwm5 = GWM(6*num_heads[0], 3*num_heads[0]*2)

        self.dsn1 = DSB( head_dim*num_heads[4] , self.amp , num_heads[4])
        self.dsn2 = DSB( head_dim*num_heads[3] , self.amp , num_heads[3])
        self.dsn3 = DSB( head_dim*num_heads[2] , self.amp , num_heads[2])
        self.dsn4 = DSB( head_dim*num_heads[1] , self.amp , num_heads[1])
        self.dsn5 = DSB( head_dim*num_heads[0] , self.amp , num_heads[0])
        

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))

    def forward(self, moving, fixed):

        # encode stage
        M1, M2, M3, M4, M5 = self.encoder(moving)
        F1, F2, F3, F4, F5 = self.encoder(fixed)

        q5, k5 = self.projblock5(F5), self.projblock5(M5)

        q5 = q5.permute(0, 3, 1, 2)
        k5 = k5.permute(0, 3, 1, 2)
        w = self.dsn5(q5, k5)
        #w = self.dsn5(w, k5)
        w = self.cwm5(w)
        flow = w

        M4 = self.transformer[3](M4, flow)
        q4,k4 = self.projblock4(F4), self.projblock4(M4)
        q4 = q4.permute(0, 3, 1, 2)
        k4 = k4.permute(0, 3, 1, 2)
        w = self.dsn4(q4, k4)
        #w = self.dsn4(w, k4)
        w = self.cwm4(w)
        flow = self.transformer[2](self.upsample_trilin(2*flow), w) + w

        M3 = self.transformer[2](M3, flow)
        q3, k3 = self.projblock3(F3), self.projblock3(M3)
        q3 = q3.permute(0, 3, 1, 2)
        k3 = k3.permute(0, 3, 1, 2)
        w = self.dsn3(q3, k3)
        #w = self.dsn3(w, k3)
        w = self.cwm3(w)
        flow = self.transformer[1](self.upsample_trilin(2 * flow), w) + w

        M2 = self.transformer[1](M2, flow)
        q2,k2 = self.projblock2(F2), self.projblock2(M2)
        q2 = q2.permute(0, 3, 1, 2)
        k2 = k2.permute(0, 3, 1, 2)
        w = self.dsn2(q2, k2)
        #w = self.dsn2(w, k2)
        w = self.cwm2(w)
        flow = self.upsample_trilin(2 *(self.transformer[1](flow, w)+w))

        M1 = self.transformer[0](M1, flow)
        q1, k1 = self.projblock1(F1), self.projblock1(M1)
        q1 = q1.permute(0, 3, 1, 2)
        k1 = k1.permute(0, 3, 1, 2)
        w = self.dsn1(q1, k1)
        #w = self.dsn1(w, k1)
        w = self.cwm2(w)
        flow = self.transformer[0](flow, w) + w

        y_moved = self.transformer[0](moving, flow)

        return y_moved, flow

if __name__ == '__main__':
    inshape = (1, 1, 80, 96, 80)
    model = DySNet_M(inshape[2:]).cuda(2)
    A = torch.ones(inshape)
    B = torch.ones(inshape)
    out, flow = model(A.cuda(2), B.cuda(2))
    print(out.shape, flow.shape)
