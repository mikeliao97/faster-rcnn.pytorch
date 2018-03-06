#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pdb 
from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN


import math
from os.path import join

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo


BatchNorm = nn.BatchNorm2d

WEB_ROOT = 'https://tigress-web.princeton.edu/~fy/dla/public_models/'



from collections import namedtuple
import json
from os.path import exists, join


Dataset = namedtuple('Dataset', ['model_hash', 'classes', 'mean', 'std',
                                 'eigval', 'eigvec', 'name'])

imagenet = Dataset(name='imagenet',
                   classes=1000,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   eigval=[55.46, 4.794, 1.148],
                   eigvec=[[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]],
                   model_hash={'dla34': '6ba26179',
                               'dla46_c': '37675969',
                               'dla46x_c': '893b8958',
                               'dla60x_c': '5de0ad33',
                               'dla60': 'e2f4df06',
                               'dla60x': '3062c917',
                               'dla102': 'd94d9790',
                               'dla102x': 'ad62be81',
                               'dla102x2': '262837b6',
                               'dla169': '0914e092'})


def get_data(data_name):
    try:
        return globals()[data_name]
    except KeyError:
        return None


def load_dataset_info(data_dir, data_name='new_data'):
    info_path = join(data_dir, 'info.json')
    if not exists(info_path):
        return None
    info = json.load(open(info_path, 'r'))
    assert 'mean' in info and 'std' in info, \
        'mean and std are required for a dataset'
    data = Dataset(name=data_name, classes=0,
                   mean=None,
                   std=None,
                   eigval=None,
                   eigvec=None,
                   model_hash=dict())
    return data._replace(**info)


def get_model_url(data, name):
    return join(WEB_ROOT, data.name,
                '{}-{}.pth'.format(name, data.model_hash[name]))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)

        return out


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Identity, self).__init__()
        assert in_channels == out_channels
        pass

    def forward(self, x):
        return x


class BasicTree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, linear_root=True, root_dim=0):
        super(BasicTree, self).__init__()
    # self.root = None
        # if levels == 0:
        #     self.tree1 = self.tree2 = None
        #     self.root = block(in_channels, out_channels, stride)
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, 1)
        else:
            self.tree1 = BasicTree(levels - 1, block, in_channels,
                                   out_channels, stride, root_dim=0)
            self.tree2 = BasicTree(levels - 1, block, out_channels,
                                   out_channels,
                                   root_dim=root_dim + out_channels)
        if levels == 1:
            # root_in_dim = out_channels * (levels + 1)
            # root_in_dim += in_channels
            root_layers = [nn.Conv2d(root_dim, out_channels,
                                     kernel_size=1, stride=1, bias=False),
                           BatchNorm(out_channels)]
            if not linear_root:
                root_layers.append(nn.ReLU(inplace=True))
            self.root = nn.Sequential(*root_layers)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        print("downsample: ", self.downsample)
        print("x.size(2)", x.size(2))
        print("x", x.size())
        pdb.set_trace()
        assert self.downsample is None or x.size(2) % 2 == 0
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(torch.cat([x1, x2, *children], 1))
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class RRoot(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 scale_residual=False):
        super(RRoot, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_residual = scale_residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.scale_residual:
            for c in children:
                if c.size(1) == x.size(1):
                    x += c
        else:
            x += children[0]
        x = self.relu(x)

        return x


class RTree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 scale_residual=False, dilation=1):
        super(RTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = RTree(levels - 1, block, in_channels, out_channels,
                               stride, root_dim=0,
                               root_kernel_size=root_kernel_size,
                               scale_residual=scale_residual,
                               dilation=dilation)
            self.tree2 = RTree(levels - 1, block, out_channels, out_channels,
                               root_dim=root_dim + out_channels,
                               root_kernel_size=root_kernel_size,
                               scale_residual=scale_residual,
                               dilation=dilation)
        if levels == 1:
            self.root = RRoot(root_dim, out_channels, root_kernel_size)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)

        linear_root = False
        if not residual_root:
            self.level2 = BasicTree(
                levels[2], block, channels[1], channels[2], 2,
                level_root=False, linear_root=linear_root)
            self.level3 = BasicTree(
                levels[3], block, channels[2], channels[3], 2,
                level_root=True, linear_root=linear_root)
            self.level4 = BasicTree(
                levels=levels[4], block=block, in_channels=channels[3], out_channels=channels[4], 
                stride=2, level_root=True, linear_root=linear_root)
            self.level5 = BasicTree(
                levels[5], block, channels[4], channels[5], 2,
                level_root=True, linear_root=linear_root)
        else:
            self.level2 = RTree(levels[2], block, channels[1], channels[2], 2,
                                level_root=False,
                                scale_residual=False)
            self.level3 = RTree(levels[3], block, channels[2], channels[3], 2,
                                level_root=True, scale_residual=False)
            self.level4 = RTree(levels[4], block, channels[3], channels[4], 2,
                                level_root=True, scale_residual=False)
            self.level5 = RTree(levels[5], block, channels[4], channels[5], 2,
                                level_root=True, scale_residual=False)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        print("forward in dla", x)
        y = []
        #Modify x with padding so it is divisible by 32 on both sides
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x

    def load_pretrained_model(self, data_name, name):
        data = imagenet
        fc = self.fc
        if self.num_classes != data.classes:
            self.fc = nn.Conv2d(
                self.channels[-1], data.classes,
                kernel_size=1, stride=1, padding=0, bias=True)
        try:
            model_url = get_model_url(data, name)
        except KeyError:
            raise ValueError(
                '{} trained on {} does not exist.'.format(data.name, name))
        self.load_state_dict(model_zoo.load_url(model_url))
        self.fc = fc


def dla34(pretrained=None, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla34')
    return model


def dla46_c(pretrained=None, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla46_c')
    return model


def dla46x_c(pretrained=None, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla46x_c')
    return model


def dla60x_c(pretrained=None, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla60x_c')
    return model


def dla60(pretrained=None, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla60')
    return model


def dla60x(pretrained=None, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla60x')
    return model


def dla102(pretrained=None, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla102')
    return model


def dla102x(pretrained=None, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla102x')
    return model


def dla102x2(pretrained=None, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla102x2')
    return model


def dla169(pretrained=None, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(pretrained, 'dla169')
    return model


class dla(_fasterRCNN):
  """Constructs a DLA-34 Model
  Args:
  pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  def __init__(self, classes, num_layers=0, pretrained=True, class_agnostic=False):
    self.model_path = 'data/pretrained_model/dla34.pth'
    self.dout_base_model = 256
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
  
    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    
    if self.pretrained == True:
      dla = dla34(pretrained='imagenet')  #default configuration
    else:
      dla = dla34(pretrained=False)
    

    #Build DLA
    '''
    #The pass through the tree asserts that size of the image is not 7x7
    #So i put two fc for the RCNN_top for prediction
    self.RCNN_base = nn.Sequential(dla.base_layer, dla.level0, dla.level1,
            dla.level2, dla.level3, dla.level4)

    self.RCNN_top = nn.Sequential(dla.level5)
    '''
    self.RCNN_base = nn.Sequential(dla.base_layer, dla.level0, dla.level1,
            dla.level2, dla.level3, dla.level4, dla.level5)
            

    self.RCNN_top = nn.Sequential(nn.Linear(512, 512), 
                                  nn.ReLU(), 
                                  nn.Linear(512, 512), 
                                  nn.ReLU())


    #TODO: Fix 512 with network specific architecture
    self.RCNN_cls_score = nn.Linear(512, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(512, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(512, 4 * self.n_classes)

    #Fix Blocks no backprop
    for i in range(len(self.RCNN_base)):
      for p in self.RCNN_base[i].parameters(): p.requires_grad=False
    
    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[4].train()
      self.RCNN_base[5].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    #This might be wrong.......
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7



    


