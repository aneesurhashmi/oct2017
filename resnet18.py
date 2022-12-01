from pickle import TRUE
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.models as models
# from models import Network2
from dataset import train_loader

from tensorboardX import SummaryWriter
# import torch.utils.tensorboard
writer = SummaryWriter()

SAVED_MODEL_NAME = "resnet18_full_e100.pt"
MODEL_PATH = f'./models/{SAVED_MODEL_NAME}'

USE_GPU = True
SHOW_SAMPLE_IMAGES = False
RUN_MODEL = True
LOAD_PREVIOUS_MODEL = False


img_h = 150
img_w = 150
num_classes = 4  # 0..9
lr = 0.01
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
print("device: ", device)
# ===================================================================================================================================== //
# ===================================================================================================================================== //


if LOAD_PREVIOUS_MODEL:
    model = torch.load(MODEL_PATH)

else:
    # model = Network2()
    # model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    model = models.resnet18()

    # examples, labels = next(iter(train_loader))
    # writer.add_graph(model, examples.reshape(-1, ))


    # Since this pretrained model is trained on ImageNet dataset, 
    # the output layers has 1000 nodes. We want to reshape this 
    # last classifier layer to fit this dataset which has 2 classes. 
    # Furthermore, in feature extracting, we don't need to calculate 
    # gradient for any layers except the last layer that we initialize. 
    # For this we need to set .requires_grad to False

    # freeze all the layers except last one
    def set_parameter_requires_grad(model, feature_extracting=True):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
                
    set_parameter_requires_grad(model)
    # Initialize new output layer
    model.fc = nn.Linear(512, num_classes)



if USE_GPU:
    model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# scheduler
# update LR with epochs
# every step_size epoch --> lr = gamma* lr 
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)


epoch_loss = []



if not RUN_MODEL:
    print("Please set run_model = True to start training. :)")
else:
    for epoch in range(epochs):
        e_loss = 0
        for idx, (images, labels) in enumerate(train_loader):
            if USE_GPU:
                images = images.to(device)
                labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()  # dl/dw
            optimizer.step()  # update w


            # if (idx+1) % 10 == 0:
            e_loss = loss.item()

            print(
                f"Epoch: {epoch+1}/{epochs}, iteration: {idx+1}, loss: {loss.item()}")
            epoch_loss.append(loss)
            # debugging
            # if idx == 400:
            #     print("exiting")
            #     break
        writer.add_scalar("Loss/train", e_loss, epoch)
    writer.flush()
    plt.plot(list(range(len(epoch_loss))), torch.tensor(epoch_loss).to("cpu"))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("resnet18_Train_loss.png")

    # torch.save(model.state_dict(), MODEL_PATH)
    torch.save(model, MODEL_PATH)
    # save model loss
    # a_file = open("./files/resnet_epoch_loss.txt", "w+")
    # for row in epoch_loss:
    #     np.savetxt(a_file, torch.tensor(row).to("cpu"))

    # a_file.close()
