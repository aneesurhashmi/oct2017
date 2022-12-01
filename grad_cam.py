# This default renderer is used for sphinx docs only. Please delete this cell in IPython.
import plotly.io as pio
pio.renderers.default = "png"

import json
import torch
from torchvision import models, transforms
from PIL import Image as PilImage

from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM

from utils import train_epoch, evaluate

# train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE = BATCH_SIZE)

PATH = '../original_data/test/CNV/CNV-53018-1.jpeg'

# Load the test image
# img = Image.open()
img = Image(PilImage.open(PATH).convert('RGB'))
# Load the class names

# with open('../data/images/imagenet_class_index.json', 'r') as read_file:
#     class_idx = json.load(read_file)
#     idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


# A ResNet Model
# device = torch.device("cuda" if torch.cuda.is_available else "cpu")
# model = torch.load(f'./models/cnn-80.pth')
# model = torch.load(f'./models/resnet50-80.pth')


model = models.resnet50(pretrained=True)
# The preprocessing model

transform = transforms.Compose([
            transforms.Resize(((150, 150))),
            transforms.ToTensor(),
        ])

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])


explainer = GradCAM(
    model=model,
    target_layer=model.layer4[-1],
    preprocess_function=preprocess
)

# print(img.shape)
# Explain the top label
explanations = explainer.explain(img)
# print(explanations)
# explanations.ipython_plot(index=0, class_names=idx2label)













# import torch
# import torch.nn as nn
# from torch.utils import data
# from torchvision.models import vgg19
# from torchvision import transforms
# from torchvision import datasets
# import matplotlib.pyplot as plt
# import numpy as np

# # use the ImageNet transformation
# transform = transforms.Compose([transforms.Resize((224, 224)), 
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# # define a 1 image dataset
# dataset = datasets.ImageFolder(root='./data/Elephant/', transform=transform)

# # define the dataloader to load that single image
# dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)


# class VGG(nn.Module):
#     def __init__(self):
#         super(VGG, self).__init__()
        
#         # get the pretrained VGG19 network
#         self.vgg = vgg19(pretrained=True)
        
#         # disect the network to access its last convolutional layer
#         self.features_conv = self.vgg.features[:36]
        
#         # get the max pool of the features stem
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
#         # get the classifier of the vgg19
#         self.classifier = self.vgg.classifier
        
#         # placeholder for the gradients
#         self.gradients = None
    
#     # hook for the gradients of the activations
#     def activations_hook(self, grad):
#         self.gradients = grad
        
#     def forward(self, x):
#         x = self.features_conv(x)
        
#         # register the hook
#         h = x.register_hook(self.activations_hook)
        
#         # apply the remaining pooling
#         x = self.max_pool(x)
#         x = x.view((1, -1))
#         x = self.classifier(x)
#         return x
    
#     # method for the gradient extraction
#     def get_activations_gradient(self):
#         return self.gradients
    
#     # method for the activation exctraction
#     def get_activations(self, x):
#         return self.features_conv(x)