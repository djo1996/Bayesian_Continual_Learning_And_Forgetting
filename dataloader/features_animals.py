#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:01:30 2024

@author: djohan
"""

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18,  ResNet18_Weights
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import numpy as np



# We prefer to load only the batch of data we are currently using into the GPU, to save GPU memory.
torch.set_default_device('cpu')

    
data_path = "../datasets/ANIMALS_FINAL/Train"
default_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(degrees=30),
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
# print(len(ds))
train_loader, test_loader = get_dataloaders(ds, [1,  0.], batch_size= 286)
# print(len(train_loader))

# class_names = ['bee', 'shark', 'gorilla', 'sparrow', 'coyote', 'mosquito', 'whale', 'orangutan', 'pigeon', 'wolf', 'fly', 'dolphin', 'chimpanzee', 'sandpiper', 'dog']

# class_names = ['chat', 'chien','papillion']
# class_names=np.loadtxt('../datasets/ANIMALS_90/name of the animals.txt',dtype=str)
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

# curr_image = inv_normalize1(image, mean, std)
# for i in range(16):
#     graph = plt.subplot(4, 4, i + 1)
#     plt.imshow(curr_image[i].numpy().astype('uint8'))
#     plt.title(label[i],fontsize=24)
#     plt.axis('off')
# plt.savefig('domain_inc_animals_a.svg')    
# plt.show()

    
# def reduce_channels(x):
#     # Split the 40 channels into 3 groups (13, 13, and 14 channels)
#     x_split = torch.split(x, [11, 11, 10], dim=1)  # Splits the channels
#     # Take the mean across each group
#     x = torch.stack([torch.mean(group, dim=1) for group in x_split], dim=1)  
#     return x  # Output shape: (16, 3, 28, 28)
 
class CNN(nn.Module):
    """
    Convolutional neural network used for classic learning on CIFAR10. 
    Same architecture as in F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
    """
    def __init__(self, ):
        super(CNN, self).__init__()
        # weights = EfficientNet_B0_Weights.DEFAULT
        weights = ResNet18_Weights
        # self.efficientnet = efficientnet_b0(weights=weights,progress=False).eval() 
        self.efficientnet = resnet18(weights=weights,progress=False).eval() 
        self.transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=15),
    ])
        modules = list(self.efficientnet.children())[:-1]  # delete the last fc layer and conv layer
        self.efficientnet = nn.Sequential(*modules)
    def forward(self, x):
        """ 
        Forward pass through the network. 
        If in training mode we apply a random horizontal flip to the input
        """
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
default_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
])

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

