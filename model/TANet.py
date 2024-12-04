# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:12:50 2023

@author: xhwch
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
from thop import profile
import time
import os
import random




class RES_Block(nn.Module):
    """Res Basic Block"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, previous_dilation=1):
        super(RES_Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation)
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + residual
        out = self.relu(out)

        return out
        


class GSA_Block(nn.Module):
    """Global Strip-wise Attention"""
    def __init__(self, inplanes, outplanes):
        super(GSA_Block, self).__init__()
        midplanes = int(outplanes//2)

        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1_h = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0))
        self.conv_1_w = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1))


        self.fuse_conv = nn.Conv2d(midplanes, midplanes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=False)
        self.conv_final = nn.Conv2d(midplanes, outplanes, kernel_size=1)

        self.mask_conv_1 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu = nn.ReLU(inplace=False)
        self.mask_conv_2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)


    def forward(self, x):
        _, _, h, w = x.size()
        x_1_h = self.pool_1_h(x)
        x_1_h = self.conv_1_h(x_1_h)
        x_1_h = x_1_h.expand(-1, -1, h, w)

        x_1_w = self.pool_1_w(x)
        x_1_w = self.conv_1_w(x_1_w)
        x_1_w = x_1_w.expand(-1, -1, h, w)

        #print("x_1_h size: ",x_1_h.shape)
        #print("x_1_w size: ",x_1_w.shape)
                
        hx = self.relu(self.fuse_conv(x_1_h + x_1_w))
        #print("h w fuse size: ",hx.shape)
        
        mask_1 = self.conv_final(hx).sigmoid()
        out1 = x * mask_1
        

        return out1



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
                
        self.RES_Block_e11 = RES_Block(32, 32)
        self.RES_Block_e12 = RES_Block(32, 32)
        self.RES_Block_e13 = RES_Block(32, 32)
        self.down_1 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.RES_Block_e21 = RES_Block(64, 64)
        self.RES_Block_e22 = RES_Block(64, 64)
        self.RES_Block_e23 = RES_Block(64, 64)
        self.down_2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)

        
    def forward(self, x):
        
        hx = self.RES_Block_e11(x)
        hx = self.RES_Block_e12(hx)
        hx = self.RES_Block_e13(hx)
        residual_1 = hx
        hx = self.down_1(hx)
        hx = self.RES_Block_e21(hx)
        hx = self.RES_Block_e22(hx)
        hx = self.RES_Block_e23(hx)
        residual_2 = hx
        hx = self.down_2(hx)
        
        return hx, residual_1, residual_2


class RESU_Block(nn.Module):
    def __init__(self, in_size, out_size):
        super(RESU_Block, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        
        self.conv_1 = nn.Conv2d(out_size*2, out_size, 1, 1, 0)
        self.relu_1 = nn.LeakyReLU(0.2, True)
        
        self.conv_block1 = RES_Block(out_size, out_size)
        self.conv_block2 = RES_Block(out_size, out_size)
        self.conv_block3 = RES_Block(out_size, out_size)
        
    def forward(self, x, bridge):
        
        #print("GSUB input size: ", x.shape)                 # [1, 128, 64, 64]
        hx = self.up(x)
        #print("GSUB up/concat input size: ", hx.shape)      # [1, 64, 128, 128]
        #print("GSUB input res size: ", bridge.shape)        # [1, 64, 128, 128]
        hx = torch.cat([hx, bridge], 1)
        #print("GSUB concat output size: ", hx.shape)        # [1, 128, 128, 128]
        hx = self.conv_1(hx)
        hx = self.relu_1(hx)
        #print("GSUB concat output resize: ", hx.shape)      # [1, 64, 128, 128]
        hx = self.conv_block1(hx)
        hx = self.conv_block2(hx)
        hx = self.conv_block3(hx)
        #print("GSUB output size: ", hx.shape)               # [1, 64, 128, 128]
        
        return hx



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.up_1 = RESU_Block(128, 64)
        self.up_2 = RESU_Block(64, 32)
        self.conv_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        
    def forward(self, x, residual_1, residual_2):
        
        hx = self.up_1(x, residual_2)
        hx = self.up_2(hx, residual_1)
        #hx = self.up_1(torch.cat((x, residual_2), dim = 1))
        #hx = self.up_2(torch.cat((hx, residual_1), dim = 1))
        
        return hx


class LPA_Block(nn.Module):
    """Local Pixel-wise Attention Block"""
    def __init__(self, kernel_size=7):
        super(LPA_Block, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x

class GDA_Block(nn.Module):
    """Global Distribution Attention Block"""
    def __init__(self, in_size, out_size):
        super(GDA_Block, self).__init__()
        
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)
        
        self.norm = nn.InstanceNorm2d(out_size//2, affine=True)


    def forward(self, x):
        out = self.conv_1(x)

        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out = out + x

        return out



class Attention_Block(nn.Module):
    def __init__(self, in_size, out_size):
        super(Attention_Block, self).__init__()
        
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        
        self.conv_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.LeakyReLU(0.2, True)
        self.conv_2 = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0)
        self.relu_2 = nn.LeakyReLU(0.2, True)
        self.conv_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu_3 = nn.LeakyReLU(0.2, True)

        self.conv_4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.relu_4 = nn.LeakyReLU(0.2, True)
        
        self.conv_4_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu_4_1 = nn.LeakyReLU(0.2, True)
        self.conv_4_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu_4_2 = nn.LeakyReLU(0.2, True)
        self.conv_4_3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.relu_4_3 = nn.LeakyReLU(0.2, True)
        
        in_split_1, in_split_2 = in_size//4, in_size//4
        out_split_1, out_split_2 = out_size//4, out_size//4
        
        
        self.GSA_Block = GSA_Block(64, 64)
        self.LPA_Block = LPA_Block()
        
        self.GDA_Block = GDA_Block(in_size, out_size)
        
        
        
    def forward(self, x):
        
        hx = self.conv_1(x)
        hx = self.relu_1(hx)
        
        hx = self.conv_4(hx)
        hx = self.relu_4(hx)
                
        hx1 = self.conv_4_1(hx)
        hx1 = self.relu_4_1(hx1)
        hx2 = self.conv_4_2(hx)
        hx2 = self.relu_4_2(hx2)
        hx3 = self.conv_4_3(hx)
        hx3 = self.relu_4_3(hx3)
        
        GSA = self.GSA_Block(hx1)
        LPA = self.LPA_Block(hx2)
        
        #print("GSA size: ", GSA.shape)
        #print("LPA size: ", LPA.shape)
        #print("split size 3: ", hx3.shape)
        
        hx = torch.cat([GSA, LPA, hx3], dim=1)
        #print("f1~3 size: ", hx.shape)
        
        hx = self.conv_2(hx)
        hx = self.relu_2(hx)
        
        hx = self.conv_3(hx)
        hx = self.relu_3(hx)
        
        hx = hx + x
        hx = self.GDA_Block(hx) + hx
        
        hx = hx + self.identity(x)
        
        return hx
        
        
        
class Attention(nn.Module):
    def __init__(self, in_size, out_size):
        super(Attention, self).__init__()
        
        self.Att_Block_1 = Attention_Block(in_size, out_size)
        self.Att_Block_2 = Attention_Block(in_size, out_size)
        self.Att_Block_3 = Attention_Block(in_size, out_size)
        self.Att_Block_4 = Attention_Block(in_size, out_size)
        self.Att_Block_5 = Attention_Block(in_size, out_size)
        self.Att_Block_6 = Attention_Block(in_size, out_size)
        self.Att_Block_7 = Attention_Block(in_size, out_size)
        self.Att_Block_8 = Attention_Block(in_size, out_size)
        self.Att_Block_9 = Attention_Block(in_size, out_size)
        self.Att_Block_10 = Attention_Block(in_size, out_size)
        
    def forward(self, x):
        
        hx = self.Att_Block_1(x)
        hx = self.Att_Block_2(hx)
        hx = self.Att_Block_3(hx)
        hx = self.Att_Block_4(hx)
        hx = self.Att_Block_5(hx)
        hx = self.Att_Block_6(hx)
        hx = self.Att_Block_7(hx)
        hx = self.Att_Block_8(hx)
        hx = self.Att_Block_9(hx)
        hx = self.Att_Block_10(hx)
        
        return hx





class TANet(nn.Module):
    """TANet"""
    def __init__(self, in_size=3 , out_size=3):
        super(TANet, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_size, 32, kernel_size=3, stride=1, padding=1)  # input
        self.relu_1 = nn.LeakyReLU(0.2, True)
        self.conv_2 = nn.Conv2d(32, out_size, kernel_size=3, stride=1, padding=1)  # output
        self.relu_2 = nn.LeakyReLU(0.2, True)
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.attention = Attention(128, 128)
        
        
        

    def forward(self, x):
        
        #print("Network input size: ", x.shape)                              # [1, 32, 256, 256] 
        hx = self.conv_1(x)
        hx = self.relu_1(hx)
        #print("Network input >> conv1 size: ", hx.shape)                    # [1, 32, 256, 256] 
        hx, residual_1, residual_2 = self.encoder(hx)
        #print("Network encoder output/decoder input size: ", hx.shape)      # [1, 128, 64, 64] 
        #print("Network decoder input res_1 size: ", residual_1.shape)       # [1, 32, 256, 256]
        #print("Network decoder input res_2 size: ", residual_2.shape)       # [1, 64, 128, 128]
        
        hx = self.attention(hx)
        
        hx = self.decoder(hx, residual_1, residual_2)
        #print("Network decoder output size: ", hx.shape)
        
        hx = self.conv_2(hx)
        hx = self.relu_2(hx)
        hx = hx + x
        #print("Network conv_2 >> output size: ", hx.shape)

        return hx



if __name__ == '__main__':
    # Debug
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = TANet().cuda()
    input = torch.randn(1, 3, 256, 256).cuda()

    with torch.no_grad():
        out = net(input)

    flops, params = profile(net, (input,))
    flops = flops / out.shape[1]
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

