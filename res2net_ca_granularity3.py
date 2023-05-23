import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as F

__all__ = ['Res2Net','res2net18']

model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}

# 坐标注意力机制
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # print("inp, out,",inp,oup)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # self.w_gran2 = nn.AdaptiveAvgPool2d((None, None))


        mip = max(8, inp // reduction)
        self.coor_conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0).cuda()
        self.coor_bn1 = nn.BatchNorm2d(mip)
        self.coor_act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0).cuda()
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0).cuda()
        self.h_swish = h_swish()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        # print("n, c, h, w",n, c, h, w)
        # x_h = self.pool_h(x)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # # 按h维cat
        # y = torch.cat([x_h, x_w], dim=2)
        # print("shape",y.shape)

        tar_w = w // 3
        # print("tar_w",tar_w)
        gran3_pre = x[:,:,:tar_w,:tar_w]
        # print("gran2_pre",gran2_pre.shape)
        x_h_pre = self.pool_h(gran3_pre)
        # print("x_h_pre", x_h_pre.shape)
        x_w_pre = self.pool_w(gran3_pre).permute(0, 1, 3, 2)
        y1 = torch.cat([x_h_pre, x_w_pre], dim=2)
        # print("y1",y1.shape)

        gran3_mid = x[:,:,tar_w:2*tar_w,tar_w:2*tar_w]
        x_h_mid = self.pool_h(gran3_mid)
        x_w_mid = self.pool_w(gran3_mid).permute(0, 1, 3, 2)
        y2 = torch.cat([x_h_mid, x_w_mid], dim=2)
        # print("y2", y2.shape)

        gran3_last = x[:,:,2*tar_w:,2*tar_w:]
        x_h_last = self.pool_h(gran3_last)
        x_w_last = self.pool_w(gran3_last).permute(0, 1, 3, 2)
        y3 = torch.cat([x_h_last, x_w_last], dim=2)
        # print("y3", y3.shape)

        y = torch.cat([y1,y2,y3], dim=2)
        # print("y1,y2,y3 shape", y.shape)

        y = self.coor_conv1(y)
        y = self.coor_bn1(y)
        y = self.coor_act(y)


        # x_h, x_w = torch.split(y, [h, w], dim=2)
        # x_w = x_w.permute(0, 1, 3, 2)
        # # a_h = self.conv_h(x_h).sigmoid()
        # # a_w = self.conv_w(x_w).sigmoid()
        # a_h = self.conv_h(x_h)
        # a_h = self.h_swish(a_h)
        # print("a_h", a_h.shape)
        # a_w = self.conv_w(x_w)
        # a_w = self.h_swish(a_w)
        # out = identity * a_w * a_h
        # print("out,",out.shape)

        # 3种粒度的第二次切片
        # 1.pre部分
        tar_w = 2 * x_h_pre.shape[2]
        tar = x_h_pre.shape[2]
        x_pre  = y[:,:, :tar_w]
        # print("x_pre", x_pre.shape)
        x_pre_h, x_pre_w  = torch.split(x_pre,[tar, tar], dim=2)
        x_pre_w = x_pre_w.permute(0, 1, 3, 2)
        # print("x_pre_h, x_pre_w",x_pre_h.shape, x_pre_w.shape)

        # 2.mid部分
        x_mid  = y[:,:, tar_w:2*tar_w]
        # print("x_mid", x_mid.shape)
        x_mid_h, x_mid_w = torch.split(x_mid, [tar, tar], dim=2)
        x_mid_w = x_mid_w.permute(0, 1, 3, 2)
        # print("x_mid_h, x_mid_w", x_mid_h.shape, x_mid_w.shape)


        # 3.last部分
        tar = x_h_last.shape[2]
        # print("tartar", tar)
        x_last = y[:,:, 2*tar_w:]
        # print("x_last", x_last.shape)
        x_last_h, x_last_w = torch.split(x_last, [tar, tar], dim=2)
        x_last_w = x_last_w.permute(0, 1, 3, 2)
        # print("x_last_h, x_last_w", x_last_h.shape, x_last_w.shape)

        # 3种粒度对应的卷积操作部分、h_swish激活函数部分
        a_pre_h = self.conv_h(x_pre_h)
        a_pre_h = self.h_swish(a_pre_h)
        a_pre_w = self.conv_w(x_pre_w)
        a_pre_w = self.h_swish(a_pre_w)
        # print("a_pre", a_pre.shape)
        a_mid_h = self.conv_h(x_mid_h)
        a_mid_h = self.h_swish(a_mid_h)
        a_mid_w = self.conv_h(x_mid_w)
        a_mid_w = self.h_swish(a_mid_w)
        # print("a_mid", a_mid.shape)
        a_last_h = self.conv_h(x_last_h)
        a_last_h = self.h_swish(a_last_h)
        a_last_w = self.conv_h(x_last_w)
        a_last_w = self.h_swish(a_last_w)
        # print("a_last", a_last.shape)

        # 对原始特征图切分成3种粒度
        tar_ori = x_h_pre.shape[2]
        identity_pre = identity[:,:,:tar_ori, :tar_ori]
        # print("identity_pre", identity_pre.shape)
        identity_mid = identity[:, :, tar_ori:2 * tar_ori, tar_ori:2 * tar_ori]
        # print("identity_mid", identity_mid.shape)
        identity_last = identity[:, :, 2 * tar_ori:, 2 * tar_ori:]
        # print("identity_last", identity_last.shape)

        out1 = identity_pre * a_pre_h * a_pre_w
        # print("out1",out1.shape )
        out1_h = self.pool_h(out1)
        out1_w = self.pool_w(out1).permute(0, 1, 3, 2)
        out1 = torch.cat([out1_h, out1_w], dim=2)

        out2 = identity_mid * a_mid_h * a_mid_w
        # print("out2",out2.shape )
        out2_h = self.pool_h(out2)
        out2_w = self.pool_w(out2).permute(0, 1, 3, 2)
        out2 = torch.cat([out2_h, out2_w], dim=2)

        out3 = identity_last * a_last_h * a_last_w
        out3_h = self.pool_h(out3)
        out3_w = self.pool_w(out3).permute(0, 1, 3, 2)
        out3 = torch.cat([out3_h, out3_w], dim=2)

        out = torch.cat([out1, out2, out3], dim = 2)
        x_h, x_w = torch.split(out, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out = x_h * x_w

        # out =  a_pre_h * a_pre_w * a_mid_h * a_mid_w * a_last_h * a_last_w
        # print("out1, out2, out3",out1.shape,out2.shape,out3.shape)
        # print("out",out.shape )

        return out



class Bottle2neck_ca(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal',
                 reduction=32):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck_ca, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        # block头部第一个1*1卷积
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        # 中间的4个3*3 卷积
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        # block尾部的1*1卷积
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.ca = CoordAtt(planes * self.expansion, planes * self.expansion)


        # 原始的一些参数
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width


    def forward(self, x):
        residual = x

        # block头部第一个1*1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("\n\nblock头部的1*1卷积处理之后： ", out.shape)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        # print("\n\nblock尾部的1*1卷积处理之前： \n\n", out.shape)
        # block尾部的1*1卷积
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # 注意力机制的forward
        out = self.ca(out)

        # 原始部分
        out += residual
        out = self.relu(out)


        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=7):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = torch.tensor(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def res2net18(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck_ca, [1, 1, 2, 1], baseWidth=26, scale=4, **kwargs)
    pretrained = False
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net18_26w_4s']))
    return model



if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net18(pretrained=False)
    model = model.cuda(0)
    # print(model(images).size())