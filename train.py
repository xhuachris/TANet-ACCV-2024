# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:29:00 2023

@author: xhwch
"""


import torch
from dataloader import Train_Loader
import random
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model.TANet import TANet
from loss import Reconstruction_criterion
import tqdm
import cv2
from tensorboardX import SummaryWriter
from joblib import cpu_count
import os
import math
import torchvision
cv2.setNumThreads(0)

def calc_psnr(result, gt):
    """
    Only calculate the first batch
    """
    result = (result + 0.5)
    gt = (gt + 0.5)
    result = result.cpu().numpy()
    gt = gt.cpu().numpy()
    mse = np.mean(np.power((result - gt), 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# training seed
seed = 666
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# hyperparameters
data_path = '/mnt/nvlab/nvlab110hua/dataset/'       # training datapath
model_name = 'TANet_exp_001'                        # exp name
crop_size = 224
start_epoch = 0
end_epoch = 500
batch_size = 16
init_lr = 1e-4
min_lr = 1e-7
net = TANet()
check_point_epoch = 100
writer = SummaryWriter(model_name)


# Traning loader
Train_set = Train_Loader(data_path, crop_size)
dataloader_train = DataLoader(Train_set, batch_size=batch_size, shuffle=True, num_workers=cpu_count(),
                              drop_last=False)

# Model and optimizer
net = nn.DataParallel(net)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=end_epoch, eta_min=min_lr)

# load pretrained
if os.path.exists('result_{}/last_{}.pth'.format(model_name, model_name)):
    print('load_pretrained')
    training_state = (torch.load('result_{}/last_{}.pth'.format(model_name, model_name)))
    start_epoch = training_state['epoch'] + 1
    new_weight = net.state_dict()
    new_weight.update(training_state['model_state'])
    net.load_state_dict(new_weight)
    new_optimizer = optimizer.state_dict()
    new_optimizer.update(training_state['optimizer_state'])
    optimizer.load_state_dict(new_optimizer)
    new_scheduler = scheduler.state_dict()
    new_scheduler.update(training_state['scheduler_state'])
    scheduler.load_state_dict(new_scheduler)

print('Start_Epoch:', start_epoch)
print('End_Epoch:', end_epoch)
for epoch in range(start_epoch, end_epoch):
    tq = tqdm.tqdm(dataloader_train, total=len(dataloader_train))
    tq.set_description(
        'Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))   
    total_train_loss = 0.
    
    total_l1_loss = 0.
    total_fft_loss = 0.

    total_train_psnr = 0.
    train_psnr_h = 0.
    train_psnr_r = 0.
    train_psnr_s = 0.
    num_h = 0
    num_r = 0
    num_s = 0
    
    for idx, sample in enumerate(tq):

        input, gt = sample['input'].to(device), sample['gt'].to(device)

        output = net(input).clamp(-0.5, 0.5)
        #print(type(output))
        net.zero_grad()
        #loss, l1_loss, fft_loss = Reconstruction_criterion(output, gt)
        loss = Reconstruction_criterion()(output, gt)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        psnr = calc_psnr(output.detach(), gt.detach())
        total_train_loss += loss.item()
        #total_l1_loss += l1_loss.item()
        #total_fft_loss += fft_loss.item()
        total_train_psnr += psnr
        
        #tq.set_postfix(L1_loss= (total_l1_loss / (idx + 1)), FFT_loss= (total_fft_loss / (idx + 1)) ,Loss=(total_train_loss / (idx + 1)), PSNR=(total_train_psnr / (idx + 1)))
        tq.set_postfix(Loss=(total_train_loss / (idx + 1)), PSNR=(total_train_psnr / (idx + 1)))
          

    scheduler.step()
    writer.add_scalar('Train_loss', total_train_loss / (idx + 1), epoch)
    writer.add_scalar('Train_psnr', total_train_psnr / (idx + 1), epoch)

    # save parameters
    scheduler_state = scheduler.state_dict()
    optimizer_state = optimizer.state_dict()
    net_state = net.state_dict()
    training_state = {'epoch': epoch, 'model_state': net_state,
                      'scheduler_state': scheduler_state, 'optimizer_state': optimizer_state}

    torch.save(training_state, 'result_{}/last_{}.pth'.format(model_name, model_name))

    if (epoch % check_point_epoch) == 0:
        torch.save(training_state, 'result_{}/epoch_{}_{}.pth'.format(model_name, epoch, model_name))
        

    if epoch == (end_epoch - 1):
        torch.save(net_state, 'result_{}/final_{}.pth'.format(model_name, model_name))
    

