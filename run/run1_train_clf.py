import argparse
import datetime
import math
import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf
sys.path.append('/data2/youngju/kimst24')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm
from utils.load_dataset import get_CUB_loaders

from torch.utils.tensorboard import SummaryWriter

import wandb

# https://www.kaggle.com/code/sharansmenon/pytorch-cubbirds200-classification


def train(args, train_loader, val_loader, test_loader, model):
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    criterion = criterion.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.96)
    summary_writer = SummaryWriter(log_dir=args.save_dir)
    best_acc = 0
    
    for e in range(args.epochs):
        model.train()
        train_loss = 0
        for i, (imgs, labels) in enumerate(tqdm(train_loader)):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            output = model(imgs)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        step_scheduler.step()
                
        total_loss = round(train_loss / len(train_loader), 4)
        summary_writer.add_scalar("train/loss", total_loss, e)
        print(f"\n ========= ⭐️ train (epoch-{e}) loss: {total_loss} ⭐️ ========= ")

        #model, _ = eval(args, train_loader, model, criterion)
        model, val_loss, acc = eval(args, val_loader, model, criterion, phase='val')
        summary_writer.add_scalar("valid/loss", val_loss, e)
        summary_writer.add_scalar("valid/acc", acc.item(), e)
        
        model, test_loss, test_acc = eval(args, test_loader, model, criterion, phase='test')
        summary_writer.add_scalar("test/loss", test_loss, e)
        summary_writer.add_scalar("test/acc", test_acc.item(), e)
        
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir,f"clf_ep-{e:02d}_lr-{args.lr}_val-acc-{acc:.4f}.pth",),
                       )
    summary_writer.close()


def eval(args, data_loader, model, criterion, phase='val'):
    model.eval()
    pred_list = []
    correct_list = []
    logit_list = []
    running_loss = 0.0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            output = model(imgs)
            preds = output.argmax(dim=1)
            running_loss += criterion(output, labels).item()

            pred_list.append(preds)
            correct_list.append(labels)
            logit_list.append(preds)

        pred_list = torch.cat(pred_list)
        correct_list = torch.cat(correct_list)
        logit_list = torch.cat(logit_list)

        total_loss = round(running_loss / len(data_loader), 4)
        acc = (pred_list == correct_list).sum().detach().cpu().numpy() / len(pred_list)
        print(print(f'\n ========= ⭐️ {phase} loss: {total_loss} acc: {round(acc*100,2)}% / ⭐️ ========= '))

    return model, total_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/data/CUB_200_2011/images")
    parser.add_argument("--save-dir", type=str, default="/data2/youngju/kimst24/outputs/clf_checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--train-ratio", type=float, default=0.7)


    args = parser.parse_args()
    
    now = datetime.datetime.now()
    save_dir = os.path.join(args.save_dir, now.strftime("%Y-%m-%d_%H%M%S"))
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    
    flags  = OmegaConf.create({})
    for k, v in vars(args).items():
        print(">>>", k, ":" , v)
        setattr(flags, k, v)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    (train_loader, val_loader, test_loader) = get_CUB_loaders(args.data_path, args.batch_size, args.train_ratio, train=True)
    # (val_loader, test_loader, valid_data_len, test_data_len) = get_CUB_loaders(args.data_path, int(args.batch_size/2), args.train_ratio, train=False)

    args.model_name = 'resnet50'
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 200)
    
    # model = models.efficientnet_b1(pretrained=True)
    # n_inputs = model.classifier[1].in_features
    # model.classifier = nn.Sequential(
    #     nn.Linear(n_inputs,2048),
    #     nn.SiLU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(2048, 200)
    # )
    
    # model = models.mobilenet_v2(pretrained=True)
    # model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=200)
    
    
    model = model.to(args.device)

    train(args, train_loader, val_loader, test_loader, model)
    OmegaConf.save(flags, os.path.join(args.save_dir, "config.yaml"))



if __name__ == "__main__":
    main()