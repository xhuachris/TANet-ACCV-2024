

import torch
from dataloader import Test_Loader, Test_Loader_h, Test_Loader_r, Test_Loader_s
import random
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model.TANet import TANet
import tqdm
import cv2
from tensorboardX import SummaryWriter
from joblib import cpu_count
import os
import math
import torchvision
import time

from torch.autograd import Variable
import torch.nn.functional as F

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


# hyperparameters
data_path = '/mnt/nvlab/nvlab110hua/dataset/'   # training datapath
model_name = 'TANet'                    # exp name
crop_size = 224
start_epoch = 0
end_epoch = 1
batch_size = 1
init_lr = 1e-4
min_lr = 1e-7
net = TANet()
check_point_epoch = 4
writer = SummaryWriter(model_name)
#iters_to_accumulate = 4    # gradient accumulation

# Traning loader
""" All/haze/rain/snow """"
Test_set = Test_Loader(data_path, crop_size)
#Test_set = Test_Loader_h(data_path, crop_size)
#Test_set = Test_Loader_r(data_path, crop_size)
#Test_set = Test_Loader_s(data_path, crop_size)

dataloader_test = DataLoader(Test_set, batch_size=batch_size, shuffle=False, num_workers=cpu_count(),
                              drop_last=False)

# Model and optimizer
net = nn.DataParallel(net)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=end_epoch, eta_min=min_lr)

# load pretrained
#if os.path.exists('result_{}/last_{}.pth'.format(model_name, model_name)):
if os.path.exists('weights/TANet.pth'):
    print('load_pretrained')
    training_state = (torch.load('weights/TANet.pth'))
    #training_state = (torch.load('result_{}/last_{}.pth'.format(model_name, model_name)))
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
    

input_h = 460
input_w = 620
for epoch in range(0,1):
    tq = tqdm.tqdm(dataloader_test, total=len(dataloader_test))
    
    total_loss = 0.
    total_psnr = 0.
    for idx, sample in enumerate(tq):
        total_test_loss = 0.
        total_test_psnr = 0.
        each_output_img = []
        each_input_img = []
        
        for i in range (0, len(sample)):
        
            # input: [B, C, H, W], gt: [B, C, H, W]
            input, gt = sample[i]['input'].to(device), sample[i]['gt'].to(device)
            name = sample[i]['name'][0]
            #print("sample ", i," input size: ", input.size())
            #print("sample ", i," gt size: ", gt.size())
            
            
            
            
            with torch.no_grad():
            
              img_tensor = input
              #img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()
              #print("sample ", i," input tensor size: ", img_tensor.size())
              
              factor = 64
              h, w = img_tensor.shape[2], img_tensor.shape[3]
              H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
              padh = H - h if h % factor != 0 else 0
              padw = W - w if w % factor != 0 else 0
              img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
              H, W = img_tensor.shape[2], img_tensor.shape[3]
              #print("sample ", i," padded input tensor size: ", img_tensor.size())
              
              input = img_tensor            
              
              output = net(input.type(torch.float32)).clamp(-0.5, 0.5)
              output = output[:,:,:h,:w]
              
              each_output_img.append(output.detach().cpu())
                
              #loss = criterion(output, gt)
              psnr = calc_psnr(output.detach(), gt.detach())
              #total_test_loss += loss.item()
              total_test_psnr += psnr
              #total_loss +=loss.item()
              total_psnr +=psnr
        #tq.set_postfix(Loss=(total_test_loss /(len(sample))), PSNR=(total_test_psnr / (len(sample))))
        tq.set_postfix(Loss=(total_test_loss /(len(sample))), PSNR=(total_test_psnr / (len(sample))))
              
              
        #final_output = output_img[:, :, 0:input_h, 0:input_w]
        
        torchvision.utils.save_image((output[0]+0.5),'./output_{}/{}'.format(model_name, name))
#print('mean_loss =', total_loss/(len(dataloader_test)*len(sample)), 'mean_psnr = ', total_psnr/(len(dataloader_test)*len(sample)) )
print('mean_psnr = ', total_psnr/(len(dataloader_test)*len(sample)) )
        
        