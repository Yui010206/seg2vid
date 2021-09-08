import torch
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models
import torch.optim as optim
import os
import logging
import torchvision.utils as tov

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))
        

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class AnoPoseFormer(nn.Module):
    def __init__(self, num_frame=8, headless=True, in_chans=2, embed_dim=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,  norm_layer=None):

        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            headless (bool): use head joints or not
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        if headless:
            num_joints = 14
        else:
            num_joints = 17

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        out_dim =  2     #### output dimension is num_joints * 2

        ### spatial patch embedding
        self.point_embed = nn.Linear(in_chans, embed_dim)
        self.type_embed = nn.Embedding(3,embed_dim)
        self.spatial_embed = nn.Embedding(num_joints+2,embed_dim)
        self.temporal_embed = nn.Embedding(num_frame,embed_dim)
        self.norm = norm_layer(embed_dim)
        self.drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.pose_head_rec = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim , out_dim)
        )

        self.pose_head_pre = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim , out_dim)
        )

        self.track_head_rec = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim , out_dim)
        )

        self.track_head_pre = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim , out_dim)
        )

    def encode_input(self, points,type_tokens,spatial_tokens,temporal_tokens):
        # points (B,T*N,2)
        # type_tokens (B,T*N)
        # spatial_tokens (B,T*N)
        # temporal_tokens (B,T*N)

        points_embedding = self.point_embed(points)
        type_embedding = self.type_embed(type_tokens)
        spatial_embedding = self.spatial_embed(spatial_tokens)
        temporal_embedding = self.temporal_embed(temporal_tokens)
        embedding = 3*points_embedding + type_embedding + spatial_embedding + temporal_embedding
        embedding = self.norm(embedding)
        embedding = self.drop(embedding)

        for blk in self.Spatial_blocks:
            embedding = blk(embedding)
            #print('blk',x.shape)

        return embedding

    def forward(self,points,type_tokens,spatial_tokens,temporal_tokens,task,headless):

        embedding = self.encode_input(points,type_tokens,spatial_tokens,temporal_tokens)
        # B, N, dim

        if task == 'MPR':

            recon_embedding = torch.truck

            if headless:
                recon_embedding = embedding[:,-14:,:].clone()
            else:
                recon_embedding = embedding[:,-17:,:].clone()

            rec = self.pose_head_rec(embedding)

            return rec

        elif task == 'MPP':

            if headless:
                predict_embedding = embedding[:,-14:,:].clone()
            else:
                predict_embedding = embedding[:,-17:,:].clone()

            pre = self.pose_head_pre(predict_embedding)

            return pre 

        elif task == 'MTR':

            rec = self.track_head_rec(embedding)

            return rec 

        elif task == 'MTP':

            if headless:
                predict_embedding = embedding[:,-16:-14,:].clone()
            else:
                predict_embedding = embedding[:,-19:-17,:].clone()

            pre = self.track_head_pre(embedding)

            return pre

        elif task == 'Rec':

            if headless:
                predict_embedding = embedding[:,-16:-14,:].clone()
            else:
                predict_embedding = embedding[:,-19:-17,:].clone()

            pre = self.track_head_pre(embedding)

            return pre

        elif task == 'Pre':

            if headless:
                predict_embedding = embedding[:,-16:-14,:].clone()
            else:
                predict_embedding = embedding[:,-19:-17,:].clone()

            pre = self.track_head_pre(embedding)

            return pre

        elif task == 'All':

            if headless:
                predict_embedding = embedding[:,-16:-14,:].clone()
            else:
                predict_embedding = embedding[:,-19:-17,:].clone()

            pre = self.track_head_pre(embedding)

            return pre



if __name__ == '__main__':
    PoseTransformer = AnoPoseFormer()
    points = torch.rand([4,16*4,2])
    type_tokens = torch.randint(0,3,(4,16*4))
    rec = PoseTransformer(points,type_tokens,type_tokens,type_tokens)

    print('rec',rec.shape)



