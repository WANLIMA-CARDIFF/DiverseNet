from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHeadV3Plus_multi_head, DeepLabHeadV3Plus_multi_head_dropout
from .utils import SimpleSegmentationModel_multi_head
from .backbone import resnet
from torch import nn
import torch
import torchvision
import re


def deeplabv3plus_multi_heads(in_number, num_classes, output_stride=16, pretrained_backbone=True):
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    backbone = resnet.__dict__['resnet50'](pretrained=pretrained_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    # backbone.conv1 = nn.Conv2d(in_number, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    inplanes = 2048
    low_level_planes = 256
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus_multi_head(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = SimpleSegmentationModel_multi_head(backbone, classifier)
    return model


def deeplabv3plus_multi_heads_dropout(in_number, num_classes, output_stride=16, pretrained_backbone=True):
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    backbone = resnet.__dict__['resnet50'](pretrained=pretrained_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    # backbone.conv1 = nn.Conv2d(in_number, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    inplanes = 2048
    low_level_planes = 256
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus_multi_head_dropout(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = SimpleSegmentationModel_multi_head(backbone, classifier)
    return model

