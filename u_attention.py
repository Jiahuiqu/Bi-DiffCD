import torch.nn as nn
import math
from timm import trunc_normal_
import einops
import torch.utils.checkpoint
import torch.nn.functional
import torch

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = 1
    h = w = int(x.shape[1] ** .5)
    # assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

        self.in_chans = dim
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(self.in_chans),
            nn.Linear(self.in_chans, self.in_chans * 4),
            Swish(),
            nn.Linear(self.in_chans * 4, self.in_chans)
        )

        self.noise_func = FeatureWiseAffine(self.in_chans, self.in_chans)

    def forward(self, x, timesteps, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, timesteps, skip)

    def _forward(self, x, timesteps, skip=None):

        t = self.noise_level_mlp(timesteps)
        x = self.noise_func(x, t)
        x = x + self.attn(self.norm1(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=256):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            noise = noise.squeeze(3).permute(0,2,1)
            x = x + noise
        return x


class U_attition(nn.Module):

    def __init__(self, img_size=224, patch_size=13, in_chans=3, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        # self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.final_layer = nn.Conv2d(self.in_chans, 3, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.conv1 = nn.Conv2d(embed_dim,self.in_chans,3,1,1)
        self.conv_dim = nn.Conv2d(256 * 2, 256, 3, 1, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}


    def forward(self, x, timesteps, y=None):
        feature = []
        x_org = x
        feature.append(x_org[:, 0:3, :, :])
        x = self.patch_embed(x)
        B, L, D = x.shape

        x = x + self.pos_embed
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)

        skips = []
        out = []

        for blk in self.in_blocks:
            x = blk(x, timesteps)
            x_temp = unpatchify(x, self.in_chans)
            out.append(x_temp)
            skips.append(x)
            x_feature = unpatchify(x, self.in_chans)
            x_feature = self.conv1(x_feature)
            x_feature = self.final_layer(x_feature)
            feature.append(x_feature)

        x = self.mid_block(x, timesteps)
        x_temp = unpatchify(x, self.in_chans)

        x_feature = unpatchify(x, self.in_chans)
        x_feature = self.conv1(x_feature)
        x_feature = self.final_layer(x_feature)
        feature.append(x_feature)

        for blk in self.out_blocks:

            x_temp = torch.cat([x_temp, out.pop()], dim=1)
            x_temp = self.conv_dim(x_temp)
            x = x_temp.reshape(x_temp.shape[0], x_temp.shape[1], x_temp.shape[2] * x_temp.shape[3]).permute(0, 2, 1)
            x = blk(x, timesteps)
            x_temp = unpatchify(x, self.in_chans)
            x_feature = unpatchify(x, self.in_chans)
            x_feature = self.conv1(x_feature)
            x_feature = self.final_layer(x_feature)
            feature.append(x_feature)

        x = self.norm(x)
        # x = self.decoder_pred(x)
        # assert x.size(1) == self.extras + L
        # x = x[:, self.extras:, :]
        assert x.size(1) == L
        x = x
        x = unpatchify(x, self.in_chans)
        x = self.conv1(x)
        x = self.final_layer(x)

        feature.append(x)

        return x, feature


class Attention_CD(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_1 = nn.Linear(dim, 256)
        self.proj_2 = nn.Linear(dim, 256)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, L, C = x.shape

        qkv_1 = self.qkv_1(x)
        qkv_2 = self.qkv_2(y)
        if ATTENTION_MODE == 'math':
            qkv_1 = einops.rearrange(qkv_1, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q_1, k_1, v_1 = qkv_1[0], qkv_1[1], qkv_1[2]  # B H L D
            qkv_2 = einops.rearrange(qkv_2, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q_2, k_2, v_2 = qkv_2[0], qkv_2[1], qkv_2[2]  # B H L D
            attn_1 = (q_1 @ k_2.transpose(-2, -1)) * self.scale
            attn_1 = (1-attn_1).softmax(dim=-1)
            attn_1 = self.attn_drop(attn_1)
            x = (attn_1 @ v_1).transpose(1, 2).reshape(B, L, C)

            attn_2 = (q_2 @ k_1.transpose(-2, -1)) * self.scale
            attn_2 = (1-attn_2).softmax(dim=-1)
            attn_2 = self.attn_drop(attn_2)
            y = (attn_2 @ v_2).transpose(1, 2).reshape(B, L, C)

        else:
            raise NotImplemented

        x = self.proj_1(x)
        y = self.proj_2(y)

        return x, y


class Block_CD(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False, patch_size=1,in_chans=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 5**2, dim))

        self.attn_CD = Attention_CD(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.attn_x = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.attn_y = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

        self.in_chans = dim
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(self.in_chans),
            nn.Linear(self.in_chans, self.in_chans * 4),
            Swish(),
            nn.Linear(self.in_chans * 4, self.in_chans)
        )

        self.patch_embed_1 = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=256)
        self.patch_embed_2 = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=256)
        # self.noise_func = FeatureWiseAffine(self.in_chans, self.in_chans)


    def forward(self, x, y):

        return self._forward(x, y)

    def _forward(self, x, y):

        x = self.patch_embed_1(x)
        y = self.patch_embed_2(y)
        x = x + self.pos_embed
        y = y + self.pos_embed
        x, y = self.attn_CD(self.norm1(x), self.norm2(y))
        out = torch.cat([x, y], dim=2)
        out = unpatchify(out, 1)

        return out


class CD_Net(nn.Module):

    def __init__(self, embed_dim=256, num_heads=1, mlp_ratio=4, qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,
                 use_checkpoint=None):
        super().__init__()

        self.CD_blocks = nn.ModuleList([
            Block_CD(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(18)])

        self.Classfy1 = nn.Sequential(
            nn.Linear(9216 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        self.flatten = nn.Flatten(1, -1)
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cuda:1")
        self.parameter = []

    def forward(self, Data_patch):

        ## U_attition中间特征
        # 0层
        f_step0_1_patch = self.CD_blocks[0](Data_patch['RGB_f_step0_1_patch'],
                                            Data_patch['SAR_f_step0_4_patch'])
        f_step0_2_patch = self.CD_blocks[1](Data_patch['RGB_f_step0_2_patch'],
                                            Data_patch['SAR_f_step0_3_patch'])
        f_step0_3_patch = self.CD_blocks[2](Data_patch['RGB_f_step0_3_patch'],
                                            Data_patch['SAR_f_step0_2_patch'])
        f_step0_4_patch = self.CD_blocks[3](Data_patch['RGB_f_step0_4_patch'],
                                            Data_patch['SAR_f_step0_1_patch'])
        # 1000层
        f_step1000_1_patch = self.CD_blocks[4](Data_patch['RGB_f_step1000_1_patch'],
                                               Data_patch['SAR_f_step1000_4_patch'])
        f_step1000_2_patch = self.CD_blocks[5](Data_patch['RGB_f_step1000_2_patch'],
                                               Data_patch['SAR_f_step1000_3_patch'])
        f_step1000_3_patch = self.CD_blocks[6](Data_patch['RGB_f_step1000_3_patch'],
                                               Data_patch['SAR_f_step1000_2_patch'])
        f_step1000_4_patch = self.CD_blocks[7](Data_patch['RGB_f_step1000_4_patch'],
                                               Data_patch['SAR_f_step1000_1_patch'])
        # 2000层
        f_step2000_1_patch = self.CD_blocks[8](Data_patch['RGB_f_step2000_1_patch'],
                                               Data_patch['SAR_f_step2000_4_patch'])
        f_step2000_2_patch = self.CD_blocks[9](Data_patch['RGB_f_step2000_2_patch'],
                                               Data_patch['SAR_f_step2000_3_patch'])
        f_step2000_3_patch = self.CD_blocks[10](Data_patch['RGB_f_step2000_3_patch'],
                                               Data_patch['SAR_f_step2000_2_patch'])
        f_step2000_4_patch = self.CD_blocks[11](Data_patch['RGB_f_step2000_4_patch'],
                                               Data_patch['SAR_f_step2000_1_patch'])

        # 输出特征
        SAR_step_0 = self.CD_blocks[12](Data_patch['SAR_f_step0_0_patch'], Data_patch['RGB_x_0_patch'])
        RGB_step_0 = self.CD_blocks[13](Data_patch['SAR_x_0_patch'], Data_patch['RGB_f_step0_0_patch'])
        SAR_step_1000 = self.CD_blocks[14](Data_patch['SAR_f_step1000_0_patch'], Data_patch['RGB_f_step1000_6_patch'])
        RGB_step_1000 = self.CD_blocks[15](Data_patch['SAR_f_step1000_6_patch'], Data_patch['RGB_f_step1000_0_patch'])
        SAR_step_2000 = self.CD_blocks[16](Data_patch['SAR_f_step2000_0_patch'], Data_patch['RGB_f_step2000_6_patch'])
        RGB_step_2000 = self.CD_blocks[17](Data_patch['SAR_f_step2000_6_patch'], Data_patch['RGB_f_step2000_0_patch'])

        feature = torch.cat([f_step0_1_patch, f_step0_2_patch, f_step0_3_patch, f_step0_4_patch, f_step1000_1_patch, f_step1000_2_patch, f_step1000_3_patch,
                   f_step1000_4_patch, f_step2000_1_patch, f_step2000_2_patch,f_step2000_3_patch, f_step2000_4_patch, SAR_step_0, RGB_step_0, SAR_step_1000,
                             RGB_step_1000, SAR_step_2000, RGB_step_2000], dim=1)

        feature = self.flatten(feature)
        RGB_step_0 = self.Classfy1(feature)

        output = RGB_step_0

        return output

