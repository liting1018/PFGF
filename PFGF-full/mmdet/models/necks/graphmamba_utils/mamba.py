import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.runner import auto_fp16
from .mamaba_module import SingleMambaBlock, LayerNorm, CrossMamba

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x
    
class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class Intra_Mamba(BaseModule):
    def __init__(self, in_channels):
        super().__init__()
        base_filter = in_channels
        self.stride=1
        self.patch_size=1
        self.embed_dim = base_filter*self.stride*self.patch_size

        self.FT_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.singlemamba = SingleMambaBlock(self.embed_dim)
        self.patchunembe = PatchUnEmbed(base_filter)

    @auto_fp16()
    def forward(self, inputs1, **kwargs):
        F_T_in = inputs1
        b, c, h, w = F_T_in.shape
        F_T = self.FT_to_token(F_T_in)
        F_T = self.singlemamba(F_T)
        F_T = self.patchunembe(F_T, (h, w))
        F_TR = F_T + F_T_in

        return F_TR
    

class Inter_Mamba(BaseModule):
    def __init__(self, in_channels):
        super().__init__()
        base_filter = in_channels
        self.stride=1
        self.patch_size=1
        self.embed_dim = base_filter*self.stride*self.patch_size

        self.FT_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.crossmamba = CrossMamba(self.embed_dim)
        self.patchunembe = PatchUnEmbed(base_filter)

    @auto_fp16()
    def forward(self, inputs1, inputs2, **kwargs):
        b, c, h, w = inputs1.shape
        F_T = self.FT_to_token(inputs1)
        F_R = self.FT_to_token(inputs2)
        F_TR = self.crossmamba(F_T, F_R)
        F_TR = self.patchunembe(F_TR, (h, w))
        return F_TR