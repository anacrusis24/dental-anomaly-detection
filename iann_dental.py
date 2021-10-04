import os
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
import cv2
import torch
import numpy as np
import random
import os
import cv2
import pandas as pd
import PIL
import pprint
import torchvision


def segment_into_quadrants(det1, shape):
    top_lines, bottom_lines, left_lines, right_lines = det1[:,1], det1[:,3], det1[:,0], det1[:,2]
    left_to_right = np.argsort(left_lines)
    top_to_bottom = np.argsort(top_lines)
    top = top_to_bottom[:len(top_to_bottom)//2]
    bottom = top_to_bottom[len(top_to_bottom)//2:]

    top_ordered = np.full(top.shape, 666)
    bottom_ordered = np.full(bottom.shape, 666)

    top_ctr = 0
    bottom_ctr = 0
    for i in left_to_right:
        if i in top:
            top_ordered[top_ctr] = int(i)
            top_ctr +=1
        elif i in bottom:
            bottom_ordered[bottom_ctr] = i
            bottom_ctr +=1
        else:
            print('there is a problem')
    m_top_left, m_bottom_left = np.mean(det1[top][:, 0]), np.mean(det1[bottom][:, 0])
    m_top_rigt, m_bottom_right = np.mean(det1[top][:, 2]), np.mean(det1[bottom][:, 2])
    m_top_top, m_bottom_top = np.mean(det1[top][:, 1]), np.mean(det1[bottom][:, 1])
    m_top_bottom, m_bottom_bottom = np.mean(det1[top][:, 3]), np.mean(det1[bottom][:, 3])
    m_top_x = int((m_top_left + m_top_rigt) / 2)
    m_bottom_x = int((m_bottom_left + m_bottom_right ) / 2)
    m_top_y = int((m_top_top + m_top_bottom) / 2)
    m_bottom_y = int((m_bottom_top + m_bottom_bottom) / 2)
    top_center_x1, top_center_y1 = m_top_x - 500, m_top_y
    bottom_center_x1, bottom_center_y1 = m_bottom_x - 500, m_bottom_y
    top_center_x2, top_center_y2 = m_top_x + 500, m_top_y
    bottom_center_x2, bottom_center_y2 = m_bottom_x + 500, m_bottom_y
    
    top_left = np.array((top_center_x1, top_center_y1))
    top_right = np.array((top_center_x2, top_center_y2))

    bottom_left = np.array((bottom_center_x1, bottom_center_y1))
    bottom_right = np.array((bottom_center_x2, bottom_center_y2))
    
    filter_top_ordered = []
    filter_bottom_ordered = []

    for i in top_ordered:
        dist1 = shortest_distance(top_left, top_right, np.array((int((np.mean(det1[i][0]) + np.mean(det1[i][2])) / 2), int((np.mean(det1[i][1]) + np.mean(det1[i][3])) / 2))))
        dist2 = shortest_distance(bottom_left, bottom_right, np.array((int((np.mean(det1[i][0]) + np.mean(det1[i][2])) / 2), int((np.mean(det1[i][1]) + np.mean(det1[i][3])) / 2))))
        
        if dist1 < dist2:
            filter_top_ordered.append(i)
        else:
            filter_bottom_ordered.append(i)
    
    for j in bottom_ordered:
        dist1 = shortest_distance(top_left, top_right, np.array((int((np.mean(det1[j][0]) + np.mean(det1[j][2])) / 2), int((np.mean(det1[j][1]) + np.mean(det1[j][3])) / 2))))
        dist2 = shortest_distance(bottom_left, bottom_right, np.array((int((np.mean(det1[j][0]) + np.mean(det1[j][2])) / 2), int((np.mean(det1[j][1]) + np.mean(det1[j][3])) / 2))))

    
        if dist1 < dist2:
            filter_top_ordered.append(j)
        else:
            filter_bottom_ordered.append(j)
    
    Q1 = []
    Q2 = []
    Q3 = []
    Q4 = []
    exclude = []
    
    for inx, i in enumerate(filter_top_ordered):
        tooth_loc = ((det1[i][1] + det1[i][3])/2)/shape[0] 
        if tooth_loc > 0.9 or tooth_loc < 0.1:
            exclude.append(i)
        for j in filter_top_ordered[inx+1:]:
            if det1[i][0] < det1[j][0] and det1[i][2] > det1[j][2] and det1[i][1] < det1[j][1] and det1[i][3] > det1[j][3]:
                exclude.append(j)
            
        tooth_center = ((det1[i][0] + det1[i][2])/2)/shape[1]
        if tooth_center < 0.5 :
            if not tooth_center < 0.1 and i not in exclude:
                Q1.append(i)
        else:
            if not tooth_center > 0.9 and i not in exclude:
                Q2.append(i)
    exclude = []        
    for inx, i in enumerate(filter_bottom_ordered):
        tooth_loc = ((det1[i][1] + det1[i][3])/2)/shape[0] 
        if tooth_loc > 0.9 or tooth_loc < 0.1:
            exclude.append(i)
        for j in filter_bottom_ordered[inx+1:]:
            if det1[i][0] < det1[j][0] and det1[i][2] > det1[j][2] and det1[i][1] < det1[j][1] and det1[i][3] > det1[j][3]:
                exclude.append(j)
        
        tooth_center = ((det1[i][0] + det1[i][2])/2)/shape[1] 
        if tooth_center < 0.5 :
            if not tooth_center < 0.1 and i not in exclude:
                Q4.append(i)
        else:
            if not tooth_center > 0.9 and i not in exclude:
                Q3.append(i)
                
    Q = np.argsort(det1[Q1, 0]).astype(int)
    Q1 = [Q1[i] for i in reversed(Q)]
    Q = np.argsort(det1[Q2, 0]).astype(int)
    Q2 = [Q2[i] for i in Q]
    Q = np.argsort(det1[Q3, 0]).astype(int)
    Q3 = [Q3[i] for i in Q]
    Q = np.argsort(det1[Q4, 0]).astype(int)
    Q4 = [Q4[i] for i in reversed(Q)]
    return Q1, Q2, Q3, Q4



def shortest_distance(p1, p2, p3):
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

def connected(det, left, right,  margin=10):
    return det[left, 2] + margin >= det[right, 0]

def includes_center(det, tooth, size, margin=10):
    return (det[tooth, 0] <= (size[1]/2) + margin) and (det[tooth, 2] + margin >= (size[1]/2))

def tooth_at_center(det, quartel, neighbor, size, quartel_on_left=True):
    
    if not len(quartel):
        return False
    center_in_quartel = includes_center(det, quartel[0], size)
    if not len(neighbor):
        center_in_neighbor = False
        they_touch = False
    else:
        center_in_neighbor = includes_center(det, neighbor[0], size)
        if quartel_on_left:
            they_touch = connected(det, quartel[0], neighbor[0])
        else:
            they_touch = connected(det, neighbor[0], quartel[0])
    
    return center_in_quartel or (center_in_neighbor and they_touch)




import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512 * block.expansion, 1)
        self.softmax = torch.nn.Sigmoid()
        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
            
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        #return F.normalize(x, dim=-1)
        return self.softmax(x)

def ResidualNet(network_type, depth, num_classes, att_type):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [5, 18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)
        
    elif depth == 5:
        model = ResNet(BasicBlock, [1, 1, 2, 1], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model


import torch
import math
import torch.nn as nn
import torch.nn.functional as F

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

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
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
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


euro_to_int = {18:1, 17:2, 16:3, 15:4, 14:5, 13:6, 12:7, 11:8,
              21:9, 22:10, 23:11, 24:12, 25:13, 26:14, 27:15, 28:16,
              38:17, 37:18, 36:19, 35:20, 34:21, 33:22, 32:23, 31:24,
              41:25, 42:26, 43:27, 44:28, 45:29, 46:30, 47:31, 48:32}
weights = 'runs/train/exp38/weights/best.pt'
device = ''
device = select_device(device)
imgsz=1024
model = attempt_load(weights, map_location=device)
model.eval()
stride = int(model.stride.max())
conf_thres=0.25
iou_thres=0.3
classes=None
agnostic_nms=False
max_det=50


# Classifier

PATH = 'cls2.pth'

model2 = ResidualNet('ImageNet', 5, 2, 'CBAM')
model2.load_state_dict(torch.load(PATH))
model2.eval()
model2.to(device)