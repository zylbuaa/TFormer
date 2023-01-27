# -*- coding： utf-8 -*-
'''
@Time: 2022/5/28 16:22
@Author:YilanZhang
@Filename:TFormer.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''

import torch
import torch.nn as nn
from models.swin_transformer.build import build_model
from models.swin_transformer.config import _C as MC
from models.swin_transformer.utils import load_pretrained
from models.model_concate_multilabel import MetaSubNet
from torch.autograd import Variable
from models.mca import CrossTransformer,SwinCrossTransformer,PatchMerging,CrossTransformer_meta
import torch.nn.functional as F

class TFormer(nn.Module):
    def __init__(self,num_classes=5):
        super(TFormer, self).__init__()
        self.cli_vit = build_model(MC)
        load_pretrained(MC,self.cli_vit)

        self.meta_subnet = MetaSubNet(20,128,128,0.3)

        self.fusion_meta = CrossTransformer_meta(x_dim=128,c_dim=128*2,depth=1,num_heads=8)

        self.fusion_block0 = SwinCrossTransformer(x_dim=192, c_dim=192, depth=1, input_resolution=[56 // 2, 56 // 2],num_heads=6)
        self.downsample_cli_0 = PatchMerging(input_resolution=[56 // 2, 56 // 2], dim=192)
        self.downsample_der_0 = PatchMerging(input_resolution=[56 // 2, 56 // 2], dim=192)
        self.fusion_block1 = SwinCrossTransformer(x_dim=384,c_dim=384,depth=1,input_resolution=[56//4,56//4],num_heads=6)
        self.downsample_cli = PatchMerging(input_resolution=[56 // 4, 56 // 4],dim=384)
        self.downsample_der = PatchMerging(input_resolution=[56 // 4, 56 // 4], dim=384)
        self.fusion_block2 = SwinCrossTransformer(x_dim=768, c_dim=768, depth=2, input_resolution=[56//8,56//8],num_heads=6)
        self.fusion_block3 = SwinCrossTransformer(x_dim=768, c_dim=768, depth=1, input_resolution=[56//8,56//8],num_heads=12)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fusion_head_cli = nn.Sequential(
            nn.Linear(768 * 1,128),
            nn.ReLU(inplace=True)
        )
        self.fusion_head_der = nn.Sequential(
            nn.Linear(768 * 1, 128),
            nn.ReLU(inplace=True)
        )

        self.num_classes = num_classes
        # DIAG fc
        hidden_num = 128*2
        self.fc_diag = nn.Linear(hidden_num,
                                 self.num_classes)  # only use for three modalities

        self.fc_pn = nn.Linear(hidden_num, 3)
        self.fc_bwn = nn.Linear(hidden_num, 2)
        self.fc_vs = nn.Linear(hidden_num, 3)
        self.fc_pig = nn.Linear(hidden_num, 3)
        self.fc_str = nn.Linear(hidden_num, 3)
        self.fc_dag = nn.Linear(hidden_num, 3)
        self.fc_rs = nn.Linear(hidden_num, 2)


    def forward(self,meta_x,cli_x,der_x):
        meta_h = self.meta_subnet(meta_x)
        # feature extraction
        cli_h = self.cli_vit.patch_embed(cli_x) #b * 3136 * 96
        cli_h = self.cli_vit.pos_drop(cli_h) #b * 3136 * 96
        cli_h = self.cli_vit.layers[0](cli_h) # b * 784 * 192

        der_h = self.cli_vit.patch_embed(der_x)  # b * 3136 * 96 der_h = self.der_vit.patch_embed(der_x)  # b * 3136 * 96
        der_h = self.cli_vit.pos_drop(der_h)  # b * 3136 * 96
        der_h = self.cli_vit.layers[0](der_h)  # b * 784 * 192

        # fusion
        cli_f0, der_f0 = self.fusion_block0(cli_h, der_h)
        cli_f0 = self.downsample_cli_0(cli_f0)
        der_f0 = self.downsample_der_0(der_f0)

        cli_h = self.cli_vit.layers[1](cli_h)  # b * 196 * 384
        der_h = self.cli_vit.layers[1](der_h)  # b * 196 * 384

        #fusion
        cli_f1, der_f1 = self.fusion_block1(cli_h + cli_f0, der_h + der_f0)
        cli_f1 = self.downsample_cli(cli_f1)
        der_f1 = self.downsample_der(der_f1)

        cli_h = self.cli_vit.layers[2](cli_h)  # b * 49 * 768
        der_h = self.cli_vit.layers[2](der_h)  # b * 49 * 768

        #fusion 1
        der_f2,cli_f2= self.fusion_block2(der_h+der_f1,cli_h+cli_f1)

        der_h = self.cli_vit.layers[3](der_h)  # b * 49 * 768
        cli_h = self.cli_vit.layers[3](cli_h) # b * 49 * 768
        der_h = self.cli_vit.norm(der_h)
        cli_h = self.cli_vit.norm(cli_h)

        #fusion 2
        der_f3,cli_f3= self.fusion_block3(der_h+der_f2,cli_h+cli_f2)

        # Cross Attention
        cli_f = self.avgpool(cli_f3.transpose(1,2))
        der_f = self.avgpool(der_f3.transpose(1,2))
        cli_f = torch.flatten(cli_f,1)
        der_f = torch.flatten(der_f,1)
        der_f = torch.cat([der_f],dim=-1)
        der_f = self.fusion_head_der(der_f)
        cli_f = torch.cat([cli_f], dim=-1)
        cli_f = self.fusion_head_cli(cli_f)

        feature_f = torch.cat([cli_f,der_f],dim=-1)

        x = self.fusion_meta(meta_h, feature_f)

        x = torch.cat([der_f, x], dim=-1)

        # fc
        diag = self.fc_diag(x)

        pn = self.fc_pn(x)
        bwv = self.fc_bwn(x)
        vs = self.fc_vs(x)
        pig = self.fc_pig(x)
        str = self.fc_str(x)
        dag = self.fc_dag(x)
        rs = self.fc_rs(x)

        return[diag,pn,bwv,vs,pig,str,dag,rs]

    def criterion(self, logit, truth,weight=None):
        if weight == None:
            loss = F.cross_entropy(logit, truth)
        else:
            loss = F.cross_entropy(logit, truth,weight=weight)

        return loss



if __name__ == '__main__':
    in_size = [20, 224, 224]
    hidden_size = [16, 64, 64]
    out_size = [1, 8, 8]
    dropouts = [0.5, 0.5, 0.5]

    model = TFormer(num_classes=5)
    print(model)

    meta = torch.randn(4, 20)
    cli = torch.randn(4, 3, 224, 224)
    der = torch.randn(4, 3, 224, 224)

    outputs = model(Variable(meta), Variable(cli), Variable(der))
    print(outputs)