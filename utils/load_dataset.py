import os
import pandas as pd

import torch
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split



def get_CUB_loaders(data_dir, batch_size, train_ratio, train = False):
        if train:
            transform = transforms.Compose([
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            all_data = datasets.ImageFolder(data_dir, transform=transform)
            train_data_len = int(len(all_data)*train_ratio)
            valid_data_len = int((len(all_data) - train_data_len)/2)
            test_data_len = int(len(all_data) - train_data_len - valid_data_len)
            train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
            return train_loader, val_loader, test_loader
        
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            all_data = datasets.ImageFolder(data_dir, transform=transform)
            train_data_len = int(len(all_data)*train_ratio)
            valid_data_len = int((len(all_data) - train_data_len)/2)
            test_data_len = int(len(all_data) - train_data_len - valid_data_len)
            train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
            return (val_loader, test_loader, valid_data_len, test_data_len)
