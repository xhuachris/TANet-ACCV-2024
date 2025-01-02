import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms
import glob
import random

class RandomRotate(object):
    def __call__(self, data):
        dirct = random.randint(0, 3)
        for key in data.keys():
            data[key] = np.rot90(data[key], dirct).copy()
        return data

class RandomFlip(object):
    def __call__(self, data):
        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.fliplr(data[key]).copy()

        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.flipud(data[key]).copy()
        return data

class RandomCrop(object):
    def __init__(self, Hsize, Wsize):
        super(RandomCrop, self).__init__()
        self.Hsize = Hsize
        self.Wsize = Wsize

    def __call__(self, data):
        H, W, C = np.shape(list(data.values())[0])
        h, w = self.Hsize, self.Wsize

        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        for key in data.keys():
            data[key] = data[key][top:top + h, left:left + w].copy()

        return data

class Normalize(object):
    def __call__(self, data):
        for key in data.keys():
            data[key] = ((data[key] / 255) - 0.5).copy()
        return data

class ToTensor(object):
    def __call__(self, data):

        for key in data.keys():
            data[key] = torch.from_numpy(data[key].transpose((2, 0, 1))).clone()
        return data

class Train_Loader(Dataset):
    def __init__(self, data_path, crop_size):
        self.input_list = []
        self.gt_list = []
        self.type_list = []
        self.style_haze = 0
        self.style_rain = 1
        self.style_snow = 2
        
        
        self.transform = transforms.Compose([RandomCrop(crop_size, crop_size), RandomFlip(), RandomRotate(), Normalize(), ToTensor()])
        
        self.input_list.extend(sorted(glob.glob(os.path.join(data_path, 'train', "haze", "in", '*'))))
        self.num_haze = len(self.input_list)
        for i in range(self.num_haze):
            self.type_list.append(self.style_haze)
        
        self.input_list.extend(sorted(glob.glob(os.path.join(data_path, 'train', "rain", "in", '*'))))
        self.num_rain = len(self.input_list) - self.num_haze 
        for i in range(self.num_rain):
            self.type_list.append(self.style_rain)
        
        self.input_list.extend(sorted(glob.glob(os.path.join(data_path, 'train', "snow", "in", '*'))))
        self.num_snow = len(self.input_list) - self.num_rain - self.num_haze
        for i in range(self.num_snow):
            self.type_list.append(self.style_snow)
        
        self.gt_list.extend(sorted(glob.glob(os.path.join(data_path, 'train', "haze", "gt", '*'))))
        self.gt_list.extend(sorted(glob.glob(os.path.join(data_path, 'train', "rain", "gt", '*'))))
        self.gt_list.extend(sorted(glob.glob(os.path.join(data_path, 'train', "snow", "gt", '*'))))

        assert len(self.input_list) == len(self.gt_list), "Missmatched Length!"

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        style = self.type_list[idx]
        #print("style:", style)
        input = cv2.imread(self.input_list[idx]).astype(np.float32)
        #print("input:", input)
        
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        
        gt = cv2.imread(self.gt_list[idx]).astype(np.float32)
        #print("gt:", gt)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        
        sample = {'input': input,
                  'gt': gt}

        if self.transform:
            sample = self.transform(sample)
        
        sample['type'] = style

        return sample

class Test_Loader(Dataset):
    def __init__(self, data_path, crop_size):
        self.input_list = []
        self.gt_list = []
        self.input_name = []
        
        '''
        self.type_list = []
        self.style_haze = 0
        self.style_rain = 1
        self.style_snow = 2
        '''

        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.input_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "haze", "in", '*'))))
        self.num_haze = len(self.input_list)
        for i in range(self.num_haze):
            self.input_name.append(self.input_list[i].replace(data_path + 'test/haze/in/',""))
        '''
        for i in range(self.num_haze):
            self.type_list.append(self.style_haze)
        '''
        
        
        self.input_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "rain", "in", '*'))))
        self.num_rain = len(self.input_list) - self.num_haze 
        for i in range(self.num_rain):
            self.input_name.append(self.input_list[i].replace(data_path + 'test/rain/in/',""))
        '''
        for i in range(self.num_rain):
            self.type_list.append(self.style_rain)
        '''

        self.input_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "snow", "in", '*'))))
        self.num_snow = len(self.input_list) - self.num_rain - self.num_haze
        for i in range(self.num_snow):
            self.input_name.append(self.input_list[i].replace(data_path + 'test/snow/in/',""))
        '''
        for i in range(self.num_snow):
            self.type_list.append(self.style_snow)
        '''

        self.gt_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "haze", "gt", '*'))))
        self.gt_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "rain", "gt", '*'))))
        self.gt_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "snow", "gt", '*'))))

        assert len(self.input_list) == len(self.gt_list), "Missmatched Length!"

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        #style = self.type_list[idx]
        name = self.input_name[idx]

        input = cv2.imread(self.input_list[idx]).astype(np.float32)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gt_list[idx]).astype(np.float32)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        sample_list = []
        sample = {'input': input,
                  'gt': gt}
        
        if self.transform:
            sample = self.transform(sample)
            #sample['type']=style
            sample['name']=name
            sample_list.append(sample)
        

        return sample_list


class Test_Loader_h(Dataset):
    def __init__(self, data_path, crop_size):
        self.input_list = []
        self.gt_list = []
        self.input_name = []
        
        '''
        self.type_list = []
        self.style_haze = 0
        self.style_rain = 1
        self.style_snow = 2
        '''

        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.input_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "haze", "in", '*'))))
        for i in range(len(self.input_list)):
            self.input_name.append(self.input_list[i].replace(data_path + 'test/haze/in/',""))

        self.gt_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "haze", "gt", '*'))))

        assert len(self.input_list) == len(self.gt_list), "Missmatched Length!"

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        name = self.input_name[idx]

        input = cv2.imread(self.input_list[idx]).astype(np.float32)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gt_list[idx]).astype(np.float32)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        sample_list = []
        sample = {'input': input,
                  'gt': gt}
        
        if self.transform:
            sample = self.transform(sample)
            sample['name']=name
            sample_list.append(sample)
        

        return sample_list


class Test_Loader_r(Dataset):
    def __init__(self, data_path, crop_size):
        self.input_list = []
        self.gt_list = []
        self.input_name = []
        
        '''
        self.type_list = []
        self.style_haze = 0
        self.style_rain = 1
        self.style_snow = 2
        '''

        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.input_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "rain", "in", '*'))))
        for i in range(len(self.input_list)):
            self.input_name.append(self.input_list[i].replace(data_path + 'test/rain/in/',""))

        self.gt_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "rain", "gt", '*'))))

        assert len(self.input_list) == len(self.gt_list), "Missmatched Length!"

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        name = self.input_name[idx]

        input = cv2.imread(self.input_list[idx]).astype(np.float32)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gt_list[idx]).astype(np.float32)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        sample_list = []
        sample = {'input': input,
                  'gt': gt}
        
        if self.transform:
            sample = self.transform(sample)
            sample['name']=name
            sample_list.append(sample)
        

        return sample_list



class Test_Loader_s(Dataset):
    def __init__(self, data_path, crop_size):
        self.input_list = []
        self.gt_list = []
        self.input_name = []
        
        '''
        self.type_list = []
        self.style_haze = 0
        self.style_rain = 1
        self.style_snow = 2
        '''

        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.input_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "snow", "in", '*'))))
        for i in range(len(self.input_list)):
            self.input_name.append(self.input_list[i].replace(data_path + 'test/snow/in/',""))

        self.gt_list.extend(sorted(glob.glob(os.path.join(data_path, 'test', "snow", "gt", '*'))))

        assert len(self.input_list) == len(self.gt_list), "Missmatched Length!"

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        name = self.input_name[idx]

        input = cv2.imread(self.input_list[idx]).astype(np.float32)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gt_list[idx]).astype(np.float32)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        sample_list = []
        sample = {'input': input,
                  'gt': gt}
        
        if self.transform:
            sample = self.transform(sample)
            sample['name']=name
            sample_list.append(sample)
        

        return sample_list