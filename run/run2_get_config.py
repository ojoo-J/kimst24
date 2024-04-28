import argparse
import datetime
import math
import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from omegaconf import OmegaConf
sys.path.append('/data2/youngju/kimst24')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, models, transforms
from utils.load_dataset import get_CUB_loaders
from utils.hook import Hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/data/CUB_200_2011/images")
    parser.add_argument("--model-path", type=str, default="/data2/youngju/kimst24/outputs/clf_checkpoints/2024-04-27_132424/clf_ep-11_lr-0.001_val-acc-0.8609.pth")
    parser.add_argument("--save-dir", type=str, default="/data2/youngju/kimst24/outputs/get_config/test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--phase", type=str, default="test")


    args = parser.parse_args()
    
    #now = datetime.datetime.now()
    #save_dir = os.path.join(args.save_dir, now.strftime("%Y-%m-%d_%H%M%S"))
    #args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    
    flags  = OmegaConf.create({})
    for k, v in vars(args).items():
        print(">>>", k, ":" , v)
        setattr(flags, k, v)
    OmegaConf.save(flags, os.path.join(args.save_dir, "config.yaml"))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


    (train_loader, train_data_len) = get_CUB_loaders(args.data_path, args.batch_size, args.train_ratio, train=True)
    (val_loader, test_loader, valid_data_len, test_data_len) = get_CUB_loaders(args.data_path, int(args.batch_size/2), args.train_ratio, train=False)
    
    if args.phase == 'train':
        target_loader = train_loader
    elif args.phase == 'val':
        target_loader = val_loader
    elif args.phase == 'test':
        target_loader = test_loader

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 200)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)
    model.eval()
    
    relu_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.ReLU):
            relu_list.append(layer)
            
    total_activation = []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(target_loader)):
            HookF = [Hook(relu_layer) for relu_layer in relu_list]
            act_list = []
            output = model(imgs.to(args.device))
            
            eq = torch.where(output.argmax(dim=1).cpu().detach() == labels,1,0)
            
            for hook in HookF:
                act_list.append(hook.outputs.cpu().detach())
                hook.clear()
                
            with open(f"{args.save_dir}/{args.phase}_activation_{i}.pkl", "wb") as fw:
                pickle.dump(act_list, fw)
            with open(f"{args.save_dir}/{args.phase}_eq_{i}.pkl", "wb") as fw:
                pickle.dump(eq, fw)
            with open(f"{args.save_dir}/{args.phase}_x_{i}.pkl", "wb") as fw:
                pickle.dump(imgs, fw)
            with open(f"{args.save_dir}/{args.phase}_y_{i}.pkl", "wb") as fw:
                pickle.dump(labels, fw)
    

if __name__ == "__main__":
    main()