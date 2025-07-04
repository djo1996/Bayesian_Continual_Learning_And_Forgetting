#!/usr/bin/env python3                     
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-4.0
#
# Code for “Bayesian continual learning and forgetting in neural networks”
# (arXiv:2504.13569)
# Portions adapted from the PyTorch project (BSD-3-Clause) 
#
# Author: Djohan Bonnet  <djohan.bonnet@gmail.com>
# Date: 2025-04-18
"""
This python file was used to create the features of the animals dataset.
"""

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18,  ResNet18_Weights
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights  ALTERNATIVE FEATURE EXTRACTOR NOT THE ONE USED FOR THE MAIN PAPER

import numpy as np

# THIS IS THE CODE USED TO CREATE THE FEATURES OF THE ANIMALS DATASET.  
# I LEAVE IT HERE FOR ANYONE WHO WANTS TO DO SOMETHING SIMILAR.  
# HOWEVER, I DO NOT PROVIDE ALL THE COMMANDS TO CREATE THE FILES USED FOR THE ANIMAL FIGURES.  
# THEY ARE SAVED IN THE DATASET FOLDER.

# We prefer to load only the batch of data we are currently using into the GPU, to save GPU memory.
torch.set_default_device('cpu')

    
data_path = "../datasets/ANIMALS_FINAL/Train"
default_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
])

def get_dataset(data_path, transform=default_transforms):
    ds = ImageFolder(data_path, transform=transform)
    return ds

def get_dataloaders(ds, lengths=[0.8, 0.2], batch_size=360, seed=42, num_workers= 0):
    train_set, test_set = random_split(ds, lengths=lengths, generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

ds = get_dataset(data_path, default_transforms)
train_loader, test_loader = get_dataloaders(ds, [1,  0.], batch_size= 286)

mean=[0.485, 0.456, 0.406] 
std=[0.229,0.224,0.225]

batch = next(iter(train_loader))
image, label = batch


plt.figure(figsize = (10, 10))


def inv_normalize1(batch_image, mean, std):
    batch_image_clone = batch_image.clone()  # n, C, H, W
    inv = transforms.Normalize(
                    mean= [-mean[i] / std[i] for i in range(len(mean))],
                    std= [1 / std[i] for i in range(len(std))]
    )
    for i in range(len(batch_image)):
        batch_image_clone[i] = inv(batch_image[i]) * 255

    return batch_image_clone.permute(0, 2, 3, 1)


 
class CNN(nn.Module):
    def __init__(self, ):
        super(CNN, self).__init__()
        weights = ResNet18_Weights
        self.efficientnet = resnet18(weights=weights,progress=False).eval() 
        self.transform_train = transforms.Compose([
    ])
        modules = list(self.efficientnet.children())[:-1]  # delete the last fc layer and conv layer
        self.efficientnet = nn.Sequential(*modules)
    def forward(self, x):
        CNN.eval()
        with torch.no_grad():
            x = self.efficientnet(x)  
            x = torch.flatten(x, 1)
            # Apply fixed channel reduction
        return x

X_train=[]
Y_train=[]
X_test=[]
Y_test=[]
CNN=CNN()
# for b in test_loader:
#     x,y=b
#     if len(x)==12:
#         X_test.append(CNN(x).detach().cpu().numpy())
#         Y_test.append(y.detach().cpu().numpy())
        
# print('test done')
# print(len(X_test))
# img = 0
# for b in train_loader:
#     x,y=b
#     X=[]
#     Y=[]
#     X.append(x)
#     Y.append(y)
#     for i in range(4):
#         X.append(CNN.transform_train(x))
#         Y.append(y)
        
#     X_train.append(CNN(torch.cat(X,axis=0)).detach().cpu().numpy())
#     Y_train.append(torch.cat(Y,axis=0).detach().cpu().numpy())
#     img+=1
#     if img%50==0:
#         print(img//50, '%')

img = 0
for b in train_loader:
    x,y=b
    X_train.append(CNN(x).detach().cpu().numpy())
    Y_train.append(y.detach().cpu().numpy())
    img+=1
    if img%50==0:
        print(img//50, '%')
        
print('train done')
print(len(X_train))

X_train=np.array(X_train)
X_train=X_train.reshape(X_train.shape[0]*X_train.shape[1],512)
Y_train=np.array(Y_train)
Y_train=Y_train.reshape(Y_train.shape[0]*Y_train.shape[1])
# X_test=np.array(X_test)
# X_test=X_test.reshape(X_test.shape[0]*X_test.shape[1],1280)
# Y_test=np.array(Y_test)
# Y_test=Y_test.reshape(Y_test.shape[0]*Y_test.shape[1])
np.savez('domain_inc_animals_train_4',X=X_train,Y=Y_train)



    
data_path = "../datasets/ANIMALS_FINAL/Test"


ds = get_dataset(data_path, default_transforms)
train_loader, test_loader = get_dataloaders(ds, [1,  0.], batch_size= 181)

 


X_train=[]
Y_train=[]
X_test=[]
Y_test=[]

img = 0
for b in train_loader:
    x,y=b
    X_train.append(CNN(x).detach().cpu().numpy())
    Y_train.append(y.detach().cpu().numpy())
    img+=1
    if img%50==0:
        print(img//50, '%')
        
print('test done')
print(len(X_train))

X_train=np.array(X_train)
X_train=X_train.reshape(X_train.shape[0]*X_train.shape[1],512)
Y_train=np.array(Y_train)
Y_train=Y_train.reshape(Y_train.shape[0]*Y_train.shape[1])
# X_test=np.array(X_test)
# X_test=X_test.reshape(X_test.shape[0]*X_test.shape[1],1280)
# Y_test=np.array(Y_test)
# Y_test=Y_test.reshape(Y_test.shape[0]*Y_test.shape[1])
np.savez('domain_inc_animals_test_4',X=X_train,Y=Y_train)

