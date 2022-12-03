from __future__ import print_function, division

import numpy as np
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from einops import rearrange
import torchvision
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torch.utils.data import  WeightedRandomSampler
from tensorboardX import SummaryWriter
from utils import train_epoch, evaluate
from dataset import get_dataloaders
import matplotlib.pyplot as plt
from collections import Counter
import time
import argparse
from vit_pytorch import ViT
from vit_pytorch import *
from tqdm import tqdm
from utils import SAMSGD

train_model = False
test_model = True
IMAGE_SIZE = 150
PATCH_SIZE = 25
EPOCHS = 30
BATCH_SIZE = 2000
ATT_HEADS = 8
LR = 0.0003
DEPTH = 8

def parse_args():
    parser = argparse.ArgumentParser(description="VIT for OCT2017",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mode", help=f"test or train mode - Default = test", default="test")
    parser.add_argument("-e", "--epochs", type=int, help=f"Number of epochs - Default = {EPOCHS}", default=EPOCHS)
    parser.add_argument("-b", "--batch_size", type=int, help=f"Batch size - Default  = {BATCH_SIZE}", default=BATCH_SIZE)
    parser.add_argument("-im", "--image_size", type=int, help=f"Shape of input image - Default = {IMAGE_SIZE}", default=IMAGE_SIZE)
    parser.add_argument("-pch", "--patch_size", type=int, help=f"Patch size for vit - Default = {PATCH_SIZE}", default=PATCH_SIZE)
    parser.add_argument("-hd", "--heads",  type=int, help=f"Number of attention heads per block- Default {ATT_HEADS}", default=ATT_HEADS)
    parser.add_argument("-lr", "--learning_rate",type=float, help=f"Learning rate - Default {LR}", default=LR)
    parser.add_argument("-d", "--depth", type=int, help=f"Number of transformer block- Default {DEPTH}", default=DEPTH)
    args = parser.parse_args()
    config = vars(args)
    return config
            

args = parse_args()
MODE = args["mode"] or args["m"]
train_model = True if MODE == "train" else False
test_model = False if MODE == "train" else True
EPOCHS = args["epochs"] or args["e"]
BATCH_SIZE = args["batch_size"] or args["b"] 
IMAGE_SIZE = args["image_size"] or args["im"]
PATCH_SIZE = args["patch_size"] or args["pch"]
ATT_HEADS = args["heads"] or args["h"]
DEPTH = args["depth"] or args["d"]
LR = args["learning_rate"] or args["lr"]


use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

torch.manual_seed(42)


# ========================== Preprocessing
# load dataloaders
loaders = get_dataloaders(BATCH_SIZE = BATCH_SIZE)
val_loader, test_loader, train_loader = loaders["val_loader"], loaders["test_loader"], loaders["train_loader"]

# # ========================== train_epoch using Sharpness-Awareness Minimization optimizer
def train_epoch(model, optimizer, data_loader, loss_history, ep, device, criterion=None, writer=None):
    total_samples = len(data_loader.dataset)
    model.train()
    total_loss = 0

    for i, (data, target) in tqdm(enumerate(data_loader)):
            data = data.to(device)
            target = target.to(device)
            output = F.log_softmax(model(data), dim=1)
            # output = output.to(device)

            # loss = F.nll_loss(output, target)
            

            def closure():
                loss =  F.nll_loss(output, target)
                loss.backward(retain_graph=True)
                return loss

            loss =  F.nll_loss(output, target)
            loss.backward(retain_graph=True)
            optimizer.step(closure)
            optimizer.zero_grad()
           
            total_loss += loss.item()
            if i % 1000 == 0:
                print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                    ' (' + '{:3.0f}'.format(100 * i * len(data) / total_samples) + '%)]  Loss: ' +
                    '{:6.4f}'.format(loss.item()))
                # loss_history.append(loss.item())
    writer.add_scalar("Vit Loss", loss.item(), ep)
    # average loss per epoch
    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)




# ======================================== #
# ======================================== #
# ============== TRAINING ================ #
# ======================================== #

if train_model == True:
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = ViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=4, channels=3,
                dim=64, depth=DEPTH, heads=ATT_HEADS, mlp_dim=128)
    model = model.to(device)
    print("Using: ", device)
    
    # tensorboard 
    writer = SummaryWriter(f'./runs/d{DEPTH}/vit-sam-{EPOCHS}')

    optimizer = SAMSGD(model.parameters(), lr=LR, rho=2.0, momentum=0.8) # using github implementation

    #optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loss_history, val_loss_history = [], []
    for epoch in range(1, EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history, epoch, device, writer=writer)
        evaluate(model, val_loader, val_loss_history, device=device)
    writer.flush()
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    torch.save(model, f'./models/d{DEPTH}/vit-sam-{PATCH_SIZE}x{IMAGE_SIZE}-{EPOCHS}.pth')
elif test_model == True:
    print("Starting test.. ")
    MODEL_NAME = f"vit-sam-{PATCH_SIZE}x{IMAGE_SIZE}-{EPOCHS}" 
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = torch.load(f"./models/d{DEPTH}/{MODEL_NAME}.pth")
    model = model.to(device)
    evaluate(model, test_loader, model_name=MODEL_NAME, device=device)