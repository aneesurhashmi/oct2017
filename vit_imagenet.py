# from datasets import load_dataset

# ds = load_dataset('beans')
# # ds
# ex = ds['train'][400]
# # ex
# image = ex['image']
# labels = ds['train'].features['labels']

# import random
# from PIL import ImageDraw, ImageFont, Image

# def show_examples(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):

#     w, h = size
#     labels = ds['train'].features['labels'].names
#     grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
#     draw = ImageDraw.Draw(grid)
#     font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 24)

#     for label_id, label in enumerate(labels):

#         # Filter the dataset by a single label, shuffle it, and grab a few samples
#         ds_slice = ds['train'].filter(lambda ex: ex['labels'] == label_id).shuffle(seed).select(range(examples_per_class))

#         # Plot this label's examples along a row
#         for i, example in enumerate(ds_slice):
#             image = example['image']
#             idx = examples_per_class * label_id + i
#             box = (idx % examples_per_class * w, idx // examples_per_class * h)
#             grid.paste(image.resize(size), box=box)
#             draw.text(box, label, (255, 255, 255), font=font)

#     return grid

# # show_examples(ds, seed=random.randint(0, 1337), examples_per_class=3)



# from transformers import ViTFeatureExtractor

# model_name_or_path = 'google/vit-base-patch16-224-in21k'
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

# # print(feature_extractor(image, return_tensors='pt').pixel_values.shape)




# # # Processing the Dataset
# # Now that you know how to read images and transform them into inputs, 
# # let's write a function that will put those two things together to 
# # process a single example from the dataset.



# ds = load_dataset('beans')

# def transform(example_batch):
#     # Take a list of PIL images and turn them to pixel values
#     inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')

#     # Don't forget to include the labels!
#     inputs['labels'] = example_batch['labels']
#     return inputs


# prepared_ds = ds.with_transform(transform)

# # prepared_ds['train'][0:2]


# # Training and Evaluation

# import torch

# def collate_fn(batch):
#     return {
#         'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
#         'labels': torch.tensor([x['labels'] for x in batch])
#     }

# import numpy as np
# from datasets import load_metric

# metric = load_metric("accuracy")
# def compute_metrics(p):
#     return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


# from transformers import ViTForImageClassification

# labels = ds['train'].features['labels'].names

# model = ViTForImageClassification.from_pretrained(
#     model_name_or_path,
#     num_labels=len(labels),
#     id2label={str(i): c for i, c in enumerate(labels)},
#     label2id={c: str(i) for i, c in enumerate(labels)}
# )


# from transformers import TrainingArguments

# training_args = TrainingArguments(
#   output_dir="./vit-base-beans",
#   per_device_train_batch_size=16,
#   evaluation_strategy="steps",
#   num_train_epochs=1,
#   fp16=True,
#   save_steps=100,
#   eval_steps=100,
#   logging_steps=10,
#   learning_rate=2e-4,
#   save_total_limit=2,
#   remove_unused_columns=False,
#   push_to_hub=False,
#   report_to='tensorboard',
#   load_best_model_at_end=True,
# )


# # from transformers import Trainer

# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     data_collator=collate_fn,
# #     compute_metrics=compute_metrics,
# #     train_dataset=prepared_ds["train"],
# #     eval_dataset=prepared_ds["validation"],
# #     tokenizer=feature_extractor,
# # )

# # def train():
# #     train_results = trainer.train()
# #     trainer.save_model()
# #     trainer.log_metrics("train", train_results.metrics)
# #     trainer.save_metrics("train", train_results.metrics)
# #     trainer.save_state()

# # def evaluate():
# #     metrics = trainer.evaluate(prepared_ds['validation'])
# #     trainer.log_metrics("eval", metrics)
# #     trainer.save_metrics("eval", metrics)

# # train()



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
from torchvision.models import vit_b_16
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

# usage: python vit.py --mode train --epochs 50 --batch_size 1000 --image_size 150 --patch_size 25 --heads 12 -lr 0.0003 --depth 12

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
loaders = get_dataloaders(BATCH_SIZE = BATCH_SIZE, IMAGE_SIZE = 224)
val_loader, test_loader, train_loader = loaders["val_loader"], loaders["test_loader"], loaders["train_loader"]

# assert len(train_loader) == 83452
# assert len(test_loader) == 1000
# assert len(val_loader) == 32


# ======================================== #
# ======================================== #
# ============== TRAINING ================ #
# ======================================== #

if train_model == True:
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model = vit_b_16(weights = "ViT_B_16_Weights.IMAGENET1K_V1")
    # model = ViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=4, channels=3,
    #             dim=64, depth=DEPTH, heads=ATT_HEADS, mlp_dim=128)
    model = model.to(device)
    print("Using: ", device)
    
    # tensorboard 
    # writer = SummaryWriter(f'./runs/vit-{EPOCHS}')
    writer = SummaryWriter(f'./runs/d{DEPTH}/vit-ft-{EPOCHS}')

    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loss_history, val_loss_history = [], []
    for epoch in range(1, EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history, epoch, device, writer=writer)
        evaluate(model, val_loader, val_loss_history, device=device, writer=writer, ep = epoch)
    writer.flush()
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    
    # torch.save(model, f'./models/d{DEPTH}/vit-{PATCH_SIZE}x{IMAGE_SIZE}-{EPOCHS}.pth')
    torch.save(model, f'./models/h{ATT_HEADS}/vit-ft-{PATCH_SIZE}x{IMAGE_SIZE}-{EPOCHS}.pth')
    
elif test_model == True:
    print("Starting test.. ")
    MODEL_NAME = f"vit-ft-{PATCH_SIZE}x{IMAGE_SIZE}-{EPOCHS}" 
    # if ATT_HEADS ==8:
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # model = torch.load(f"./models/d{DEPTH}/{MODEL_NAME}.pth")
    model = torch.load(f"./models/h{ATT_HEADS}/{MODEL_NAME}.pth")
    model = model.to(device)
    # else:
    #     device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    #     model = torch.load(f"./models/d{DEPTH}/{MODEL_NAME}.pth")
    #     model = model.to(device)

    evaluate(model, test_loader, model_name=MODEL_NAME, device=device)
    