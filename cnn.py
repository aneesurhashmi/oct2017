from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from collections import Counter
import time
import argparse
from utils import train_epoch, evaluate
from dataset import get_dataloaders

# usage: python cnn.py --mode train --epochs 50 --batch_size 1000 --image_size 150 --patch_size 25 --heads 12 -lr 0.0003 --depth 12

train_model = False
test_model = True
EPOCHS = 30
BATCH_SIZE = 2000
LR = 0.0003

def parse_args():
    parser = argparse.ArgumentParser(description="cnn-v2 for OCT2017",
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

# print(len(train_loader))
# print(len(test_loader))
# print(len(val_loader))

# assert len(train_loader) == "83452"
# assert len(test_loader) == 1000
# assert len(val_loader) == 32

class CNN (nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.empty_cache()
        # n_out = (n_in + 2*padding - kernel_size)/stride + 1
        # 396 is the out size here (size_in = 400)
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)  # 196
        self.conv3 = nn.Conv2d(64, 128, 3)  # 192
        self.conv4 = nn.Conv2d(128, 128, 3)  # 192
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*7*7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, 4)

    def forward(self, input):
        # print(f'Input shape: {input.shape}')
        out = self.pool(F.relu(self.conv1(input)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = self.pool(F.relu(self.conv4(out)))
        out = out.view(-1, 128*7*7)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

# ======================================== #
# ======================================== #
# ============== TRAINING ================ #
# ======================================== #
# ======================================== #

if train_model == True:
    print(f"Starting training:")
    print(f"Total epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    # print("...")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = CNN().to(device)
    print("Using: ", device)
    # tensorboard 
    writer = SummaryWriter(f'cnn-{EPOCHS}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loss_history, val_loss_history = [], []
    for epoch in range(1, EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history, epoch, device, criterion=criterion, writer=writer)
        evaluate(model, val_loader, val_loss_history, device=device, criterion=criterion)
    writer.flush()
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    torch.save(model, f'./models/cnn-{EPOCHS}.pth')
elif test_model == True:
    print("Starting test")
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = CNN().to(device)
    print(f'./saved_models/cnn-{EPOCHS}.pth')
    MODEL_NAME = f"cnn-{EPOCHS}" 
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    criterion = nn.CrossEntropyLoss()
    model = torch.load(f'./models/cnn-{EPOCHS}.pth')
    model = model.to(device)
    evaluate(model, test_loader, model_name=MODEL_NAME, device=device, criterion=criterion)