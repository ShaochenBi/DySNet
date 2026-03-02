'''
Xmorpher

Original code retrieved from:
https://github.com/Solemoon/XMorpher

Original paper:
Shi, Jiacheng, et al. "Xmorpher: Full transformer for deformable medical image registration via cross attention." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2022.

Modified and tested by:
Shaochen Bi
bisc0507@163.com
HKUST
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath
import matplotlib.pyplot as plt
import numpy as np
import einops

class Mlp(nn.Module):
    """ Multilayer perceptron."""

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

class DySA(nn.Module):
    """
    AdSB is included in DySA.
    This is the 3×3 version of DySNet. The single-point version of DySNet can be found in DySNet_M.
    Args:
        input_dim (int): Number of input channels.
        amp (int): Attention window radius.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: False
        q_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
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
            nn.Conv2d(input_dim, 9 * 2, kernel_size=1, bias=True), 
            )
        
    def forward_cross(self, x_kv, x_q):
        """
        AdSB is included in DySA.
        Args:
            x_q: query features with shape (B, C, H, W)
            x_kv: key/value features with shape (B, C, H, W)
        Returns:
            x_v: output features with shape (B, C, H, W)
        """
        B, C, H, W = x_q.shape
        q = self.q(x_q).reshape(B, C // self.num_heads, self.num_heads, H, W)
        kv = self.kv(x_kv).reshape(B, 2, C // self.num_heads, self.num_heads, H, W)
        k = kv[:, 0] # (B, C//num_heads, num_heads, H, W)
        v = kv[:, 1] # (B, C//num_heads, num_heads, H, W)

        q = q * self.scale

        offset_input = torch.cat([x_q, x_kv], dim=1)  # (B, 2*C, H, W)

        offset = self.offset_predictor(offset_input)  # (B, 18, H, W),

        offset = offset.reshape(B, 9, 2, H, W)

        # offset: (B, 9, 2, H, W)
        B, num_offsets, _, H, W = offset.shape
        _, C_head, num_heads, _, _ = k.shape

        grid = offset.permute(0, 1, 3, 4, 2).contiguous()  # (B, num_offsets, H, W, 2)

        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1 

        k_exp = k.unsqueeze(2).expand(-1, -1, num_offsets, -1, -1, -1)  # (B, C_head, num_offsets, num_heads, H, W)
        v_exp = v.unsqueeze(2).expand(-1, -1, num_offsets, -1, -1, -1)

        k_reshape = k_exp.permute(0, 3, 2, 1, 4, 5).reshape(B * num_heads * num_offsets, C_head, H, W)
        v_reshape = v_exp.permute(0, 3, 2, 1, 4, 5).reshape(B * num_heads * num_offsets, C_head, H, W)

        grid = grid.unsqueeze(2).repeat(1, 1, num_heads, 1, 1, 1)  # (B, num_offsets, num_heads, H, W, 2)

        grid = grid.permute(0, 2, 1, 3, 4, 5).reshape(B * num_heads * num_offsets, H, W, 2)

        k_sampled = F.grid_sample(k_reshape, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        v_sampled = F.grid_sample(v_reshape, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        k_sampled = k_sampled.view(B, num_heads, num_offsets, C_head, H, W)
        v_sampled = v_sampled.view(B, num_heads, num_offsets, C_head, H, W)

        k_sampled = k_sampled.permute(0, 3, 2, 1, 4, 5).contiguous()
        v_sampled = v_sampled.permute(0, 3, 2, 1, 4, 5).contiguous()
        
        pad_dim = (self.amp, self.amp, self.amp, self.amp)
        k_padded = F.pad(k_sampled, pad_dim, "constant")
        v_padded = F.pad(v_sampled, pad_dim, "constant")

        attn_list = []
        idx = 0

        for i in range(2 * self.amp + 1):
            for j in range(2 * self.amp + 1):
                
                k_slice = k_padded[:, :, idx, :, :, :]  # (B, C_head, num_heads, H_pad, W_pad)
                k_patch = k_slice[:, :, :, i:i + H, j:j + W]  # (B, C_head, num_heads, H, W)

                attn_ij = torch.sum(q * k_patch, dim=1, keepdim=True)  # (B,1,num_heads,H,W)
                attn_list.append(attn_ij)

                idx += 1

        attn = torch.cat(attn_list, dim=1)  # (B, num_offsets, num_heads, H, W)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        v_ = 0
        idx = 0
        for i in range(2 * self.amp + 1):
            for j in range(2 * self.amp + 1):
                v_slice = v_padded[:, :, idx, :, :, :] 
                v_patch = v_slice[:, :, :, i:i + H, j:j + W]  # (B, C_head, num_heads, H, W)

                attn_slice = attn[:, idx:idx + 1, :, :, :]  

                v_ = v_ + attn_slice * v_patch  # (B, C_head, num_heads, H, W)

                idx += 1

        v_ = v_.permute(0, 1, 2, 3, 4).contiguous()  # (B, C_head, num_heads, H, W)
        v_ = v_.reshape(B, C_head * num_heads, H, W)  # (B, C, H, W)

        x_v = self.proj(v_)
        x_v = self.proj_drop(x_v)
        return x_v

    def forward(self, x_q, x_kv=None, mode='cross'):
        """
        Forward function.
        Args:
            x_q: input features with shape (B, C, H, W)
            x_kv: input features with shape (B, C, H, W), optional for cross-attention
        Returns:
            x_v: output features with shape (B, C, H, W)
        """
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
        x_v = self.attn(x_q, x_kv, mode=mode).contiguous()
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

    
class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class PointMaxMerging(nn.Module):
    """ Point Max Pooling Merging Layer for 2D input
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.GroupNorm
    """

    def __init__(self, dim, norm_layer=nn.GroupNorm):
        super().__init__()
        self.dim = dim
        self.linear = nn.Conv2d(dim, 2 * dim, kernel_size=1, stride=1, bias=False)
        self.norm = norm_layer(1, dim)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: Input feature, tensor size (B, C, H, W)
        """
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.max_pool(x)
        x = self.norm(x)
        x = self.linear(x)
        return x

class PointAvgUpExpand(nn.Module):
    def __init__(self, dim, norm_layer=nn.GroupNorm):
        super().__init__()
        self.dim = dim
        self.up_conv = nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, bias=False)
        self.norm = norm_layer(1, dim)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x = self.norm(x)
        x = self.up_conv(x)
        x = self.up(x)
        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage (2D version).
    Args:
        amp (int): amplification coefficient for searching window
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add bias to query/key/value. Default: False
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float or list, optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.GroupNorm
        resample (nn.Module or None, optional): Downsample or Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 dim,
                 amp,
                 depth,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.GroupNorm,
                 resample=None,
                 use_checkpoint=False):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            DSB(
                dim=dim,
                amp=amp,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        self.resample = None
        if resample is not None:
            self.resample = resample(dim=dim, norm_layer=norm_layer)

    def forward(self, x_A, x_B, mode='cross'):
        if mode == 'self':
            for blk in self.blocks:
                if self.use_checkpoint:
                    x_A = checkpoint.checkpoint(blk, x_A, None, mode=mode)
                else:
                    x_A = blk(x_A, None, mode=mode)
            return x_A
        else:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x_A = checkpoint.checkpoint(blk, x_A, x_B)
                    x_B = checkpoint.checkpoint(blk, x_B, x_A)
                else:
                    x_A = blk(x_A, x_B)
                    x_B = blk(x_B, x_A)
            return x_A, x_B


class PatchEmbedOverlap(nn.Module):
    def __init__(self, patch_size=(3, 3), in_chans=1, embed_dim=96, norm_layer=nn.GroupNorm):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, padding=1)
        self.norm = norm_layer(1, embed_dim)

    def forward(self, x):
        """
        Forward function.
        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            tensor of shape (B, embed_dim, H_out, W_out)
        """
        x = self.proj(x)
        x = self.norm(x)
        return x

class DySNet_X(nn.Module):
    """
       structure: 4 encoding stages(BasicLayer) + 4 decoding stages(BasicLayerUp)
    """
    def __init__(self,
                 amp=1,
                 in_chans=1,
                 out_chans=2,
                 embed_dim=24,
                 depths=[2, 2, 4, 2],
                 num_heads=[1, 2, 4, 8],
                 #num_heads=[1, 2, 4, 8, 16],
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.GroupNorm,
                 use_checkpoint=True,
                 mode='cross',
                 encoder_only=False,
                 DS=False):
        super().__init__()
        self.mode = mode
        self.encoder_only = encoder_only
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.DS = DS
        self.point_embed = PatchEmbedOverlap(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # build layers
        self.down_layers = nn.ModuleList()
        self.pool_layer = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                amp=amp,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)

            self.down_layers.append(layer)
            if i_layer < self.num_layers - 1:
                pool = PointMaxMerging(int(embed_dim * 2 ** i_layer), norm_layer=norm_layer)
                self.pool_layer.append(pool)

        self.up_layers = nn.ModuleList()
        self.uppool_layer = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.Point_fusion = nn.ModuleList()
        self.Point_revising = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i_layer in reversed(range(self.num_layers)):
            concat_linear = nn.Conv2d(2 * int(embed_dim * 2 ** i_layer), int(embed_dim * 2 ** i_layer), kernel_size=1)
            up_layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                amp=amp,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            if i_layer > 0:
                uppool = PointAvgUpExpand(int(embed_dim * 2 ** i_layer), norm_layer=norm_layer)
                self.uppool_layer.append(uppool)
            self.up_layers.append(up_layer)
            self.concat_back_dim.append(concat_linear)


            self.Point_fusion = nn.Sequential(
                    nn.Conv2d(2 * int(embed_dim * 2 ** i_layer), self.embed_dim, kernel_size=1),
                    nn.LeakyReLU()
                )

            self.Point_revising.append(nn.Conv2d(embed_dim, out_chans, kernel_size=1))
            self.norm.append(norm_layer(1, int(embed_dim * 2 ** i_layer)))

        if self.encoder_only:
            self.classifier = nn.Linear(self.num_layers, out_chans, bias=True)
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward_cross_encoder(self, img_A, img_B):
        f_A = self.point_embed(img_A)
        f_B = self.point_embed(img_B)
        features_A = []
        features_B = []
        for i, layer in enumerate(self.down_layers):
            f_A_org, f_B_org = layer(f_A, f_B)
            if i < len(self.down_layers) - 1:
                f_A = self.pool_layer[i](f_A_org)
                f_B = self.pool_layer[i](f_B_org)
            features_A.append(f_A_org)
            features_B.append(f_B_org)
        return features_A, features_B

    def forward_cross_decoder(self, features_A, features_B):
        f_A_list = []
        f_B_list = []
        for inx, layer_up in enumerate(self.up_layers):
            if inx == 0:
                f_A, f_B = layer_up(features_A[-inx - 1], features_B[-inx - 1])
                f_A_list.append(f_A)
                f_B_list.append(f_B)
                f_A = self.uppool_layer[inx](f_A)
                f_B = self.uppool_layer[inx](f_B)
            else:
                f_A = torch.cat([f_A, features_A[-inx - 1]], dim=1)
                f_B = torch.cat([f_B, features_B[-inx - 1]], dim=1)
                f_A = self.concat_back_dim[inx](f_A)
                f_B = self.concat_back_dim[inx](f_B)
                f_A, f_B = layer_up(f_A, f_B)
                f_A_list.append(f_A)
                f_B_list.append(f_B)
                if inx < len(self.up_layers) - 1:
                    f_A = self.uppool_layer[inx](f_A)
                    f_B = self.uppool_layer[inx](f_B)   

        f_A2B = self.Point_revising[-1](self.Point_fusion[-1](self.norm[-1](f_A_list[-1])))
        f_B2A = self.Point_revising[-1](self.Point_fusion[-1](self.norm[-1](f_B_list[-1])))

        return f_A2B, f_B2A

    def forward(self, img_A, img_B=None):

        features_A, features_B = self.forward_cross_encoder(img_A, img_B)
        f_A, f_B = self.forward_cross_decoder(features_A, features_B)
        return f_A, f_B