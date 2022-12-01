from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
from tensorboardX import SummaryWriter
import time
import argparse
from utils import train_epoch, evaluate
from dataset import get_dataloaders

# usage: python oct.py --mode train --epochs 50 --batch_size 1000 --image_size 150 --patch_size 25 --heads 12 -lr 0.0003 --depth 12

train_model = False
test_model = True
# IMAGE_SIZE = 150
# PATCH_SIZE = 25
EPOCHS = 30
BATCH_SIZE = 2000
LR = 0.0003

def parse_args():
    parser = argparse.ArgumentParser(description="mobilenet-v2 for OCT2017",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mode", help=f"test or train mode - Default = test", default="test")
    parser.add_argument("-e", "--epochs", type=int, help=f"Number of epochs - Default = {EPOCHS}", default=EPOCHS)
    parser.add_argument("-b", "--batch_size", type=int, help=f"Batch size - Default  = {BATCH_SIZE}", default=BATCH_SIZE)
    parser.add_argument("-lr", "--learning_rate",type=float, help=f"Learning rate - Default {LR}", default=LR)
    args = parser.parse_args()
    config = vars(args)
    return config
            

args = parse_args()
MODE = args["mode"] or args["m"]
train_model = True if MODE == "train" else False
test_model = False if MODE == "train" else True
EPOCHS = args["epochs"] or args["e"]
BATCH_SIZE = args["batch_size"] or args["b"]
LR = args["learning_rate"] or args["lr"]

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

torch.manual_seed(42)

# load dataloaders
loaders = get_dataloaders(BATCH_SIZE = BATCH_SIZE)
val_loader, test_loader, train_loader = loaders["val_loader"], loaders["test_loader"], loaders["train_loader"]

# assert len(train_loader) == 83452
# assert len(test_loader) == 1000
# assert len(val_loader) == 32


# ======================================== #
# ======================================== #
# ============== TRAINING ================ #
# ======================================== #
# ======================================== #

if train_model == True:
    print(f"Starting training:")
    print(f"Total epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    # print(f', Image Size {IMAGE_SIZE}, Patch size: {PATCH_SIZE}')
    print("...")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    for param in model.parameters():
        param.requires_grad = False            
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    model = model.to(device)
   
    print("Using: ", device)
    # tensorboard 
    writer = SummaryWriter(f'mobilenetv2-{EPOCHS}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loss_history, val_loss_history = [], []
    for epoch in range(1, EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history, epoch, device, criterion=criterion, writer=writer)
        evaluate(model, val_loader, val_loss_history, device=device, criterion=criterion)
    writer.flush()
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    torch.save(model, f'./models/mobilenetv2-{EPOCHS}.pth')
elif test_model == True:
    print("Starting test...")
    MODEL_NAME = f"mobilenetv2-{EPOCHS}" 
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    criterion = nn.CrossEntropyLoss()
    model = torch.load(f'./models/mobilenetv2-{EPOCHS}.pth')
    model = model.to(device)
    evaluate(model, test_loader, model_name=MODEL_NAME, device=device, criterion=criterion)