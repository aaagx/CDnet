# Global imports
import copy
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops

import torch.nn as nn
import torch.fft
import torchvision as tv
import torchvision
from torchvision import datasets, models, transforms
from torch.nn.modules.utils import _pair
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from mmcv.ops.carafe import CARAFEPack

## Types
from typing import Dict
from torch import Tensor
import sys
sys.path.append("./src")
from models.convnextV2 import ConvNeXtV2,load_convnextV2
from models.inceptionNext import inceptionnext_base
from losses.gem import GeM
from models.stegFormer import RestormerBlock
from models.stegFormer import to_4d
from models.stegFormer import to_3d


# Legacy resnet50 backbone
class OldBackbone(nn.Sequential):
    def __init__(self, resnet):
        super(OldBackbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(OldBackbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


# Legacy resnet50 head
class OldRes5Head(nn.Sequential):
    def __init__(self, resnet):
        super(OldRes5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.featmap_names = ['feat_res4', 'feat_res5']
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(OldRes5Head, self).forward(x)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


# Generic Backbone
class Backbone(nn.Module):
    def forward(self, x):
        y = self.body(x)
        return y


# Generic Head
class Head(nn.Module):
    def forward(self, x) -> Dict[str, Tensor]:
        feat = self.head(x)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return {"feat_res4": x, "feat_res5": feat}


# Resnet Backbone
class ResnetBackbone(Backbone):
    def __init__(self, resnet):
        super().__init__()
        return_layers = {
            'layer3': 'feat_res4',
        }
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)
        self.out_channels = 1024


# Resnet Head
class ResnetHead(Head):
    def __init__(self, resnet):
        super().__init__()
        self.head = resnet.layer4
        self.out_channels = [1024, 2048]
        self.featmap_names = ['feat_res4', 'feat_res5']


# Convnext Backbone
class ConvnextBackbone(Backbone):
    def __init__(self, convnext):
        super().__init__()
        return_layers = {
            '5': 'feat_res4',

        }
        self.body = IntermediateLayerGetter(convnext.features, return_layers=return_layers)
        self.out_channels = convnext.features[5][-1].block[5].out_features

class fpnBackbone(Backbone):
    def __init__(self,fpn):
        super().__init__()
        return_layers = {
            'downsample3': 'feat_res4',
            'downsample4': 'feat_res3',
            'downsample2': 'feat_res2',
        }
        self.body= IntermediateLayerGetter(fpn, return_layers=return_layers)
        self.out_channels = 512
    def forward(self, x):
        y = self.body(x)
        return y


class fpnHead(Head):
    def __init__(self, fpn):
        super().__init__()
        self.head = nn.Sequential(
            fpn.downsample4,
            fpn.layer4,
        )
        self.out_channels = [
            512,#512
            1024,#1024
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']

    def forward(self, x) -> Dict[str, Tensor]:
        feat = self.head(x)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return {"feat_res4": x, "feat_res5": feat} 

#FPN的类，
class fpnHead2(Head):
    def __init__(self,convnext):
        super().__init__()
        self.downsample=copy.deepcopy(convnext.features[6])
        self.layer = copy.deepcopy(convnext.features[7])#512 to 1024 3
        #对C5减少通道数得到P5
        self.toplayer = nn.Conv2d(1024,512,1,1,0)
        #3x3卷积融合特征
        self.smooth1 = nn.Conv2d(512, 512, 3, 1, 1)
        #横向连接，保证通道数相同
        self.latlayer1 = nn.Conv2d(512,512,1,1,0)
        self.featmap_names = ['feat_res4', 'feat_res5']
        self.out_channels = [512,1024]
        
    # def forward(self, x) -> Dict[str, Tensor]:
    def forward(self, x):

        # print(x.shape) 1042 512 14 14
        c5 = self.layer(self.downsample(x).contiguous())#1024
        #自上而下
        p5 = self.toplayer(c5)#1024 512 256
        p4 = self._upsample_add(p5.contiguous(), self.latlayer1(x.contiguous()))
        feat=self.smooth1(p4)
        # x = torch.amax(x, dim=(2, 3), keepdim=True)
        # feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        # print(feat.shape) torch.Size([1024, 1024, 1, 1])

        # return {"feat_res4": x, "feat_res5": feat}
        # print(feat.shape)
        return feat

    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        # print(y.shape)
        return F.interpolate(x,size=(H,W),mode='bilinear',align_corners=False).contiguous() + y
    

# Convnext Head
class ConvnextV2Head(Head):
    def __init__(self, convnext):
        super().__init__()
        self.head = nn.Sequential(
            convnext.downsample_layers[3],
            convnext.stages[3],
        )
        self.out_channels = [
            convnext.stages[2][-1].pwconv2.out_features,
            convnext.stages[3][-1].pwconv2.out_features,
        ]
        print(self.out_channels)
        self.featmap_names = ['feat_res4', 'feat_res5']

class ConvnextV2Backbone(Backbone):
    def __init__(self, convnext):
        super().__init__()
        return_layers = {
            '5': 'feat_res4',
        }
        self.body= IntermediateLayerGetter(nn.Sequential(
            convnext.downsample_layers[0],
            convnext.stages[0],
            convnext.downsample_layers[1],
            convnext.stages[1],
            convnext.downsample_layers[2],
            convnext.stages[2],
        ), return_layers=return_layers)
        self.out_channels = convnext.stages[2][-1].pwconv2.out_features

class inceptionHead(Head):
    def __init__(self, convnext):
        super().__init__()
        self.head = nn.Sequential(
            convnext.stages[3],
        )
        self.out_channels = [
            512,
            1024
        ]
        print(self.out_channels)
        self.featmap_names = ['feat_res4', 'feat_res5']

class inceptionBackbone(Backbone):
    def __init__(self, convnext):
        super().__init__()
        return_layers = {
            '3': 'feat_res4',
        }
        self.body= IntermediateLayerGetter(
                    nn.Sequential(convnext.stem
                                ,convnext.stages[0]
                                ,convnext.stages[1]
                                ,convnext.stages[2]),return_layers)
        self.out_channels = 512
        
# Convnext Head
class ConvnextHead(Head):
    def __init__(self, convnext):
        super().__init__()
        self.head = nn.Sequential(
            convnext.features[6],
            convnext.features[7],
        )
        self.out_channels = [
            convnext.features[5][-1].block[5].out_features,#512
            convnext.features[7][-1].block[5].out_features,#1024
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']


class ConvnextHead_gem(Head):
    def __init__(self, convnext):
        super().__init__()
        self.head = nn.Sequential(
            convnext.features[6],
            convnext.features[7],
        )
        self.out_channels = [
            convnext.features[5][-1].block[5].out_features,#512
            convnext.features[7][-1].block[5].out_features,#1024
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']
        self.GeMpool=GeM()


    def forward(self, x) -> Dict[str, Tensor]:
        feat = self.head(x)
        x = self.GeMpool(x)
        feat = self.GeMpool(feat)
        # print(x)
        # print(feat)
        return {"feat_res4": x, "feat_res5": feat}



            
# resnet model builder function
def build_resnet(arch='resnet50', pretrained=True,
        freeze_backbone_batchnorm=True, freeze_layer1=True,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d, user_arm_module=False):
    # weights
    if pretrained:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None

    head4=None
    head2=None
    head3=None
    # load model
    if freeze_backbone_batchnorm:
        resnet = torchvision.models.resnet50(weights=weights, norm_layer=norm_layer)
    else:
        resnet = torchvision.models.resnet50(weights=weights)

    # freeze first layers
    resnet.conv1.requires_grad_(False)
    resnet.bn1.requires_grad_(False)
    if freeze_layer1:
        resnet.layer1.requires_grad_(False)

    if user_arm_module==False:
    # setup backbone architecture
        backbone, head = ResnetBackbone(resnet), ResnetHead(resnet)
    else:
        backbone, head = ResnetBackbone(resnet), ArmRes5Head(resnet)
        head2=ResnetHead(resnet)
        head3=ArmRes5Head(resnet)
        head4=ArmRes5Head(resnet)
       
    # return backbone, head

    return backbone, head,head2,head3,head4


# convnext model builder function
def build_convnext(arch='convnext_base', pretrained=True, freeze_layer1=True, user_arm_module=False,use_gem=False):
    # weights
    weights = None
    head=None
    # load model
    if arch == 'convnext_tiny':
        print('==> Backbone: ConvNext Tiny')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_tiny(weights=weights)
    elif arch == 'convnext_small':
        print('==> Backbone: ConvNext Small')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_small(weights=weights)
    elif arch == 'convnext_base':
        print('==> Backbone: ConvNext Base')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_base(weights=weights)
    elif arch == 'convnext_large':
        print('==> Backbone: ConvNext Large')
        if pretrained:
            weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_large(weights=weights)
    elif arch == 'convnextV2_base':
        print('==> Backbone: ConvNextV2 base')
        convnext = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        if pretrained:
            load_convnextV2("/home/ubuntu/GFN-1.1.0/checkpoints/convnextv2_base_22k_224_ema.pt",convnext)
            print("/home/ubuntu/GFN-1.1.0/checkpoints/convnextv2_base_22k_224_ema.pt")
    elif arch == 'convnext_fpn':
         print('==> Backbone: convnext_fpn')
         if pretrained:
            weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
         convnext = torchvision.models.convnext_base(weights=weights)
    elif arch == 'inceptionNext':
        if pretrained:
            print('==> Backbone: inceptionNext pretrained')
            convnext=inceptionnext_base(True)
    else:
        raise NotImplementedError

    
    # freeze first layer
    head3=None
    head2=None
    head4=None
    fpn=None
    if freeze_layer1:
        if arch == 'convnextV2_base':
            convnext.downsample_layers.requires_grad_(False)
            backbone, head = ConvnextV2Backbone(convnext), ConvnextV2Head(convnext)
        elif arch == 'inceptionNext':
            convnext.stem.requires_grad_(False)
            backbone, head=inceptionBackbone(convnext),inceptionHead(convnext)
        else:
            if user_arm_module == False:
                convnext.features[0].requires_grad_(False)
                backbone= ConvnextBackbone(convnext)
                if use_gem :
                    print("use gem head to detect")
                    head= ConvnextHead_gem(convnext)
                    print("use arm head to get re_id feature")
                    head3=ArmNextHead(convnext,use_gem=False)
                    print("use decouple head to two stage detect")
                    head4=copy.deepcopy(head)
                else:
                    print("use normal head to detect")
                    head=ConvnextHead(convnext)
                    print('use arm head to get re_id feature')
                    head3=ArmNextHead(convnext)
                    print('use noraml head to get scene feature')
                    head4=None
                if arch == 'convnext_fpn':
                    # fpn=FPN(convnext)
                    head2=fpnHead2(convnext)
                

            else:
                convnext.features[0].requires_grad_(False)
                backbone, head = ConvnextBackbone(convnext), ArmNextHead(convnext)

    # return backbone, head
    return backbone, head , head2, head3,head4,fpn





class ArmRes5Head(Head):
    def __init__(self, resnet,image_size=14):
        super().__init__()
        self.head = copy.deepcopy(resnet.layer4)
        self.out_channels = [1024, 2048]
        self.featmap_names = ['feat_res4', 'feat_res5']
        self.mlP_model = ARM_Mixer(in_channels=256, image_size=image_size, patch_size=1)
        self.qconv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.qconv2 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1)
        
    def forward(self, x) -> Dict[str, Tensor]:
        #print(x.shape)
        qconv1 = self.qconv1(x)
        x_sc_mlp_feat=self.mlP_model(qconv1)
        qconv2 = self.qconv2(x_sc_mlp_feat)
        x=qconv2
        feat = self.head(x)
        x = torch.amax(x, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return {"feat_res4": x, "feat_res5": feat}

class ArmNextHead(Head):
    def __init__(self, convnext,image_size=14,patch_size=1,use_gem=False):
        super().__init__()
        self.head = nn.Sequential(
            copy.deepcopy(convnext.features[6]),
            copy.deepcopy(convnext.features[7]),
        )
        self.out_channels = [
            convnext.features[5][-1].block[5].out_features,
            convnext.features[7][-1].block[5].out_features,
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']
        
        self.mlP_model = ARM_Mixer(in_channels=256, image_size=image_size, patch_size=1)
        self.qconv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.qconv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        if use_gem:
            self.GeMpool=GeM()
            self.use_gem=True
        else:
            self.use_gem=False

        
    def forward(self, x) -> Dict[str, Tensor]:
        #print(x.shape)
        # x=self.inconv(x)
        qconv1 = self.qconv1(x)
        x_sc_mlp_feat=self.mlP_model(qconv1)
        qconv2 = self.qconv2(x_sc_mlp_feat)
        # qconv2=self.outconv(qconv2)
        feat = self.head(qconv2)
        if self.use_gem:
            qconv2 = self.GeMpool(qconv2)
            feat = self.GeMpool(feat) 
        else:
            qconv2 = torch.amax(qconv2, dim=(2, 3), keepdim=True)
            feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return {"feat_res4": qconv2, "feat_res5": feat}




class SceneNextHead(Head):
    def __init__(self, convnext):
        super().__init__()
        self.head = nn.Sequential(
            copy.deepcopy(convnext.features[6]),
            copy.deepcopy(convnext.features[7]),
        )
        self.out_channels = [
            convnext.features[5][-1].block[5].out_features,
            convnext.features[7][-1].block[5].out_features,
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']
        
        self.mlP_model = ARM_Mixer(in_channels=256, image_size=56, patch_size=1)
        self.qconv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.qconv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        
    def forward(self, x) -> Dict[str, Tensor]:
        #print(x.shape)
        # x=self.inconv(x)
        qconv1 = self.qconv1(x)
        x_sc_mlp_feat=self.mlP_model(qconv1)
        qconv2 = self.qconv2(x_sc_mlp_feat)
        # qconv2=self.outconv(qconv2)
        feat = self.head(qconv2)
        qconv2 = torch.amax(qconv2, dim=(2, 3), keepdim=True)
        feat = torch.amax(feat, dim=(2, 3), keepdim=True)
        return {"feat_res4": qconv2, "feat_res5": feat}
        

# PS_ARM
class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_features, image_size ,num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_patches, expansion_factor, dropout)
        self.image_size = image_size        
        self.spatial_att = SpatialGate()

    def SpatialGate_forward(self, x):
        residual =x 
        BB, HH_WW, CC = x.shape
        HH =  WW = int(math.sqrt(HH_WW))
        x = x.reshape(BB, CC, HH, WW)
        x = self.spatial_att(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(BB, -1, CC)
        x = residual + x    
        return x

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x_pre_norm = self.SpatialGate_forward((x))
        x = self.norm(x_pre_norm)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, image_size, num_patches, expansion_factor, dropout,use_CA=False):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, expansion_factor, dropout)
        self.image_size =image_size
        self.use_CA=use_CA
        if(use_CA):
            self.CA=RestormerBlock(dim=256,num_heads=32,ffn_expansion_factor=2,bias=True,LayerNorm_type='WithBias')
        else:
            self.channel_att = ChannelGate(num_features, )


    def ChannelGate_forward(self, x):
        residual =x 
        BB, HH_WW, CC = x.shape
        HH =  WW = int(math.sqrt(HH_WW))       
        x = x.reshape(BB, CC, HH, WW)
        x = self.channel_att(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(BB, -1, CC)
        x = residual + x    
        return x

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        if(self.use_CA==False):
            x_pre_norm = self.ChannelGate_forward(x)
        else:
            x_pre_norm = self.CA(to_4d(x,14,14))
            x_pre_norm=to_3d(x_pre_norm)
                     
        x = self.norm(x_pre_norm)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, image_size, num_patches, expansion_factor, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(
            num_patches, image_size, num_features, expansion_factor, dropout
        )
        self.channel_mixer = ChannelMixer(
            num_patches, image_size, num_features, expansion_factor, dropout
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_features)
        return x


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class ARM_Mixer(nn.Module):
    def __init__(
        self,
        image_size=14,
        patch_size=1,
        in_channels=256,
        num_features=256,
        expansion_factor=2,        
        dropout=0.5,
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        self.mixers = MixerLayer(num_patches, image_size, num_features, expansion_factor, dropout)
        self.simam = SimAM()

    def forward(self, x):
        residual = x
        BB, CC, HH, WW= x.shape
        patches = x.permute(0, 2, 3, 1)
        patches = patches.view(BB, -1, CC)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mixers(patches)
        
        embedding_rearrange = embedding.reshape(BB, CC,HH,WW)
        embedding_final = embedding_rearrange + self.simam(x)+x
        return embedding_final




#FPN的类，
class FPN(nn.Module):
    def __init__(self,convnext):
        super(FPN,self).__init__()
        self.inplanes = 256
        self.out_channels = convnext.features[5][-1].block[5].out_features
        
        self.alpha1=nn.Parameter(torch.full((1,), 0.5)).float()
        self.alpha2=nn.Parameter(torch.full((1,), 0.5)).float()
        self.alpha3=nn.Parameter(torch.full((1,), 0.5)).float()

        #处理输入的C1模块
        self.inputLayer=convnext.features[0]#3 to 128
        self.hiddenChannel=128
        #搭建自上而下的C2、C3、C4、C5
        self.layer1 = convnext.features[1]#128 to 128 3
        self.downsample2=convnext.features[2]
        self.layer2 = convnext.features[3]#128 to 256 3
        self.downsample3=convnext.features[4]
        self.layer3 = convnext.features[5]#256 to 512 27
        self.downsample4=convnext.features[6]
        self.layer4 = convnext.features[7]#512 to 1024 3
        

        #对C5减少通道数得到P5
        self.toplayer = nn.Conv2d(1024,self.hiddenChannel,1,1,0)\
        #carafe上采样
        self.upsample1=CARAFEPack(self.hiddenChannel,scale_factor=2)
        self.upsample2=CARAFEPack(self.hiddenChannel,scale_factor=2)
        self.upsample3=CARAFEPack(self.hiddenChannel,scale_factor=2)
        #3x3卷积融合特征
        self.smooth1 = nn.Conv2d(self.hiddenChannel, 512, 3, 1, 1)
        self.smooth2 = nn.Conv2d(self.hiddenChannel, 512, 3, 1, 1)
        self.smooth3 = nn.Conv2d(self.hiddenChannel, 512, 3, 1, 1)


        #横向连接，保证通道数相同
        self.latlayer1 = nn.Conv2d(512,self.hiddenChannel,1,1,0)
        self.latlayer2 = nn.Conv2d(256, self.hiddenChannel, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(128, self.hiddenChannel, 1, 1, 0)

    def upsample(self, x,upsample):
        model = upsample.cuda()
        x = x.cuda()
        out = model(x)
        return out
    
    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.interpolate(x,size=(H,W),mode='bilinear',align_corners=False).contiguous() + y

    def forward(self,x):

        # 自下而上
        c1 = self.inputLayer(x)#128
        c2 = self.layer1(c1)#128
        c3 = self.layer2(self.downsample2(c2))#256
        c4 = self.layer3(self.downsample3(c3))#512
        c5 = self.layer4(self.downsample4(c4))#1024

        #自上而下
        p5 = self.toplayer(c5)#1024 512 256
        # p4 = self.upsample(p5,self.upsample1).contiguous()+self.latlayer1(c4) 
        # p3 = self.upsample(p4,self.upsample2).contiguous()+self.latlayer2(c3)  
        # p2 = self.upsample(p3,self.upsample3).contiguous()+self.latlayer3(c2) 
        p4 = self.upsample(p5,self.upsample1).contiguous()
        p3 = self.upsample(p4,self.upsample2).contiguous()
        p2 = self.upsample(p3,self.upsample3).contiguous()


 
        #卷积融合，平滑处理
        p4 = self.smooth1(p4)*self.alpha1+self.smooth1(self.latlayer1(c4))*(1-self.alpha1) 
        p3 = self.smooth2(p3)*self.alpha2+self.smooth2(self.latlayer2(c3))*(1-self.alpha2)  
        p2 = self.smooth3(p2)*self.alpha3+self.smooth3(self.latlayer3(c2))*(1-self.alpha3) 
        
        return {'feat_res4':c4,'p4':p4,'p3':p3,'p2':p2}