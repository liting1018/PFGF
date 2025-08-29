import torch
import torch.nn as nn
import torch.nn.functional as F
from .ConvGRU import ConvGRUCell
from .mamba import Inter_Mamba, Intra_Mamba


class PPM(nn.Module):
    def __init__(self, chnn_in, rd_sc, dila):
        super(PPM, self).__init__()
        chnn = chnn_in // rd_sc
        convs = [nn.Sequential(
            nn.Conv2d(chnn_in, chnn, 3, padding=ii, dilation=ii, bias=False),
            nn.BatchNorm2d(chnn),
            nn.ReLU(inplace=True))
            for ii in dila]
        self.convs = nn.ModuleList(convs)

    def forward(self, inputs):
        feats = []
        for conv in self.convs:
            feat = conv(inputs)
            feats.append(feat)
        return feats


class GraphReasoning(nn.Module):
    def __init__(self, chnn_in, rd_sc, dila, n_iter, num_inter_graphmambas, num_intra_graphmambas):
        super().__init__()
        self.n_iter = n_iter
        self.ppm_rgb = PPM(chnn_in, rd_sc, dila)
        self.ppm_z = PPM(chnn_in, rd_sc, dila)
        self.n_node = len(dila)
        self.graph_rgb = GraphModel(self.n_node, chnn_in//rd_sc)
        chnn = chnn_in * 2 // rd_sc
        C_ca = [nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(chnn // 4, chnn // 2, 1, bias=False))
            for ii in range(2)]
        self.C_ca = nn.ModuleList(C_ca)
        C_pa = [nn.Conv2d(chnn_in//rd_sc, 1, 1, bias=False) for ii in range(2)]
        self.C_pa = nn.ModuleList(C_pa)
        # self.inter_mamba = Inter_Mamba(chnn_in//rd_sc)#就一个Inter_Mamba块
        
        # self.intra_mamba = nn.ModuleList()
        # for i in range(self.n_node):
        #     self.intra_mamba.append(Intra_Mamba(chnn_in // rd_sc))#每个节点使用不同的 Intra_Mamba 实例

        # 初始化多个 Inter_Mamba和Intra_Mamba块，不共享参数，不同节点之间不共享参数，但每个节点经过相同数量的 Intra_Mamba 块
        # self.inter_mambas = nn.ModuleList(
        #     [Inter_Mamba(chnn_in // rd_sc) for _ in range(num_inter_graphmambas)]
        # )
        # self.intra_mambas = nn.ModuleList()
        # for i in range(self.n_node):
        #     intra_mamba_blocks = [Intra_Mamba(chnn_in // rd_sc) for _ in range(num_intra_graphmambas)]
        #     self.intra_mambas.append(nn.Sequential(*intra_mamba_blocks))

        # 初始化多个 Inter_Mamba和Intra_Mamba块，所有节点共享
        self.inter_mambas = nn.ModuleList([Inter_Mamba(chnn_in // rd_sc) for _ in range(num_inter_graphmambas)])
        self.intra_mambas = nn.ModuleList([Intra_Mamba(chnn_in // rd_sc) for _ in range(num_intra_graphmambas)])
    def _enh(self, Func, src, dst):
        out = torch.sigmoid(Func(src)) * dst + dst
        return out

    def _inn(self, Func, feat):
        feat = [fm.unsqueeze(1) for fm in feat]
        feat = torch.cat(feat, 1)
        for ii in range(self.n_iter):#两层GNN
            feat = Func(feat)
        feat = torch.split(feat, 1, 1)
        feat = [fm.squeeze(1) for fm in feat]
        return feat

    # def _int(self, Func, src_1, src_2):#原作者的做法
    #     out_1 = src_2 * torch.sigmoid(Func[1](src_2 - src_1)) + src_1
    #     out_2 = src_1 * torch.sigmoid(Func[0](src_1 - src_2)) + src_2
    #     return out_1, out_2

    # def _int(self, src_1, src_2):
    #     out_1 = self.inter_mamba(src_1, src_2)#生成式特征结点和判别式特征结点更新后，再进行一个inter_mamba块
    #     return out_1

    def _int(self, src_1, src_2):
        for inter_mamba in self.inter_mambas:#多个inter_mamba块
            out_1 = inter_mamba(src_1, src_2)
        return out_1
    
    def forward(self, inputs, ii, node=False):
        feat_rgb, z, nd_rgb = inputs
        feat_rgb = self.ppm_rgb(feat_rgb)#构造结点：将每一个尺度的特征图分成三个不同尺度的特征图作为初始结点
        if node:#到第二个尺度的特征图就不再构造结点
            feat_rgb = [self._enh(self.C_ca[0], nd_rgb, fm) for fm in feat_rgb]
        if ii == 0:
            z = self.ppm_z(z)
            z = [F.interpolate(item, size=(feat_rgb[0].size(2), feat_rgb[0].size(3)), mode='bilinear', align_corners=False) for item in z]
            for j in range(self.n_node):
                feat_rgb[j] = self._int(feat_rgb[j], z[j])#生成式特征和判别式特征之间的交互
                # feat_rgb[j] = self.inter_mamba(feat_rgb[j], z[j])#生成式特征结点和判别式特征结点更新后，再进行inter_mamba
        feat_rgb = self._inn(self.graph_rgb, feat_rgb)#同一模态不同尺度的特征结点之间构造边及消息传递，结点更新等
        
        # for j in range(self.n_node):#判别式结点更新完，进行一个intra_mamba块,不共享
        #     feat_rgb[j] = self.intra_mamba[j](feat_rgb[j])

        # 每个节点经过相同数量的 Intra_Mamba 块，不同节点之间不共享参数
        # for j in range(self.n_node):
        #     feat_rgb[j] = self.intra_mambas[j](feat_rgb[j])

        for j in range(self.n_node):  # 判别式结点更新完，进行多个 intra_mamba，参数共享
            for intra_mamba in self.intra_mambas:
                feat_rgb[j] = intra_mamba(feat_rgb[j])
        return feat_rgb


class GraphModel(nn.Module):
    def  __init__(self, N, chnn_in=256):
        super().__init__()
        self.n_node = N
        chnn = chnn_in
        self.C_wgt = nn.Conv2d(chnn*(N-1), (N-1), 1, groups=(N-1), bias=False)
        self.ConvGRU = ConvGRUCell(chnn, chnn, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        b, n, c, h, w = inputs.shape
        feat_s = [inputs[:,ii,:] for ii in range(self.n_node)]
        pred_s =[]
        for idx_node in range(self.n_node):
            h_t = feat_s[idx_node]
            h_t_m = h_t.repeat(1, self.n_node-1, 1, 1)
            h_n = torch.cat([feat_s[ii] for ii in range(self.n_node) if ii != idx_node], dim=1)
            msg = self._get_msg(h_t_m, h_n)#消息传递
            m_t = torch.sum(msg.view(b, -1, c, h, w), dim=1)#每个节点都会接收来自其他节点的消息，这里是对这些消息进行聚合，以更新自己的状态
            h_t = self.ConvGRU(m_t, h_t)#更新自己的状态，及特征表示，用到了门控机制
            base = feat_s[idx_node]
            pred_s.append(h_t*self.gamma+base)
        pred = torch.stack(pred_s).permute(1, 0, 2, 3, 4).contiguous()
        return pred

    def _get_msg(self, x1, x2):
        b, c, h, w = x1.shape
        wgt = self.C_wgt(x1 - x2).unsqueeze(1).repeat(1, c//(self.n_node-1), 1, 1, 1).view(b, c, h, w)
        out = x2 * torch.sigmoid(wgt)
        return out#从其他结点到当前阶段的加权消息
    

class CGR(nn.Module):
    def __init__(self, n_iter=1, chnn_side=(256, 512, 1024), chnn_targ=(256, 512, 1024), rd_sc=32, dila=(64, 32, 16), num_inter_graphmambas=1, num_intra_graphmambas=1):
        super().__init__()
        self.n_graph = len(chnn_side)
        n_node = len(dila)
        graph = [GraphReasoning(ii, rd_sc, dila, n_iter, num_inter_graphmambas, num_intra_graphmambas) for ii in chnn_side]
        self.graph = nn.ModuleList(graph)
        C_cat = [nn.Sequential(
            nn.Conv2d(ii//rd_sc*n_node, ii//rd_sc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ii//rd_sc),
            nn.ReLU(inplace=True))
            for ii in (chnn_side+chnn_side)]
        self.C_cat = nn.ModuleList(C_cat)
        idx = [ii for ii in range(len(chnn_side))]
        C_up = [nn.Sequential(
            nn.Conv2d(chnn_side[ii] // rd_sc, chnn_targ[ii], 1, 1, 0, bias=False),
            nn.BatchNorm2d(chnn_targ[ii]),
            nn.ReLU(inplace=True))
            for ii in (idx)]
        self.C_up = nn.ModuleList(C_up)

    def forward(self, inputs):
        img, z = inputs  
        nd_rgb, nd_key = None, False
        graphouts = []
        for ii in range(self.n_graph):#遍历三个不同尺度特征图，对每个尺度特征图构造图
            feat_rgb = self.graph[ii]([img[ii], z, nd_rgb], ii, nd_key)
            feat_rgb = torch.cat(feat_rgb, 1)
            feat_rgb = self.C_cat[ii](feat_rgb)
            nd_rgb, nd_key = feat_rgb, True
            cas_rgb = self.C_up[ii](nd_rgb)
            cas_rgb = cas_rgb + img[ii]#残差连接
            graphouts.append(cas_rgb)
        return graphouts
