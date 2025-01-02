# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:26:41 2023

@author: xhwch
"""


import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        #print(diff)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        
        return loss


class Reconstruction_criterion(nn.Module):
    """ L1 loss + FFT loss """

    def __init__(self, alpha = 3e-2):
        super(Reconstruction_criterion, self).__init__()
        self.alpha = 3e-2
    
    def forward(self, x, y):
        l1_loss = CharbonnierLoss()(x, y)
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft = torch.stack((x_fft.real, x_fft.imag), -1)
        #x_fft = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)
    
        y_fft = torch.fft.fft2(y, dim=(-2, -1))
        y_fft = torch.stack((y_fft.real, y_fft.imag), -1)
        #y_fft = torch.rfft(y, signal_ndim=2, normalized=False, onesided=False)

        fft_loss = nn.L1Loss()(x_fft, y_fft)

        reconstruction_loss = l1_loss + self.alpha * fft_loss

        return reconstruction_loss #, l1_loss, 0.03 * fft_loss

