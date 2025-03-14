#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:36:55 2024

@author: djohan
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
from mnist import MNIST
import pickle

# We prefer to load only the batch of data we are currently using into the GPU, to save GPU memory.
torch.set_default_device('cpu')

class DataLoader(object):
    """Object used to load different dataset, plot images"""

    def __init__(self, args_dict):
        super().__init__()
        self.dataset_name = args_dict['dataset']  # name of dataset cifar10/cifar100/animals/MNIST/KMNIST
        self.input_transformation = args_dict['input_transformation']
        self.discretize_levels = args_dict['discretize_levels']
        self.shift = args_dict['shift']
        self.scale = args_dict['scale']
        self.archi_fcnn = args_dict['archi_fcnn']
        self.num_classes = args_dict['num_classes']
        valid_names = {'MNIST', 'KMNIST', 'CIFAR10', 'CIFAR100', 'CIFAR110','CIFAR20','ANIMALS'}
        if self.dataset_name  not in valid_names:
            raise ValueError(
                "Invalid name {!r}, should be one of {}".format(
                    self.dataset_name, valid_names))
        valid_transforms = {'normalize','standardize', 'None'}
        if self.input_transformation  not in valid_transforms:
            raise ValueError(
                "Invalid name {!r}, should be one of {}".format(
                    self.input_transformation, valid_transforms))
        self.X_ood=None
        self.X_ood_P=None
        self.load_ood()
        self.load_all()
       
            
    def load_all(self, ):
        """return the entire train and test dataset, for animals it is extracted features"""
        if self.dataset_name == 'MNIST':
            # Load MNIST dataset
            mndata = MNIST('./datasets/mnistdata/')
            X_train, Y_train = mndata.load_training()
            X_test, Y_test = mndata.load_testing()
            self.X_train = torch.tensor(X_train, dtype=torch.float32)/255
            self.Y_train = torch.tensor(Y_train, dtype=torch.long)
            self.X_test = torch.tensor(X_test, dtype=torch.float32)/255
            self.Y_test = torch.tensor(Y_test, dtype=torch.long)
            if self.input_transformation=='normalize':
                self.normalize()
            elif self.input_transformation=='standardize':
                self.standardize()
            else : 
                print('be carefull at your input distribution')
            self.shift_and_scale(self.shift, self.scale)
            if self.discretize_levels>0:
                self.discretize(self.discretize_levels)
        
        if self.dataset_name == 'CIFAR10':
            # Load CIFAR10 dataset
            X_train,Y_train=np.load('./datasets/train_cifar.npz')['X'],np.load('./datasets/train_cifar.npz')['Y']
            X_test,Y_test=np.load('./datasets/test_cifar.npz')['X'],np.load('./datasets/test_cifar.npz')['Y']
            self.X_train = torch.tensor(X_train, dtype=torch.float32)/255
            self.Y_train = torch.tensor(Y_train, dtype=torch.long)
            self.X_test = torch.tensor(X_test, dtype=torch.float32)/255
            self.Y_test = torch.tensor(Y_test, dtype=torch.long)
        
        if self.dataset_name == 'CIFAR110':
            # Load CIFAR10 dataset
            X_train,Y_train=np.load('./datasets/train_cifar.npz')['X'],np.load('./datasets/train_cifar.npz')['Y']
            X_test,Y_test=np.load('./datasets/test_cifar.npz')['X'],np.load('./datasets/test_cifar.npz')['Y']
            X_train = torch.tensor(X_train, dtype=torch.float32)/255
            Y_train = torch.tensor(Y_train, dtype=torch.long)
            X_test = torch.tensor(X_test, dtype=torch.float32)/255
            Y_test = torch.tensor(Y_test, dtype=torch.long)

            X_train_tasks = []
            Y_train_tasks = []
            X_test_tasks = []
            Y_test_tasks = []  
            num_task = 10//self.num_classes
            for i in range(num_task):
                X_train_tasks.append(X_train[(Y_train > i*self.num_classes -1) & (Y_train < (i+1)*self.num_classes)])
                X_test_tasks.append(X_test[(Y_test > i*self.num_classes -1) & (Y_test < (i+1)*self.num_classes)])
                Y_train_tasks.append(Y_train[(Y_train > i*self.num_classes -1) & (Y_train < (i+1)*self.num_classes)]-i*self.num_classes)
                Y_test_tasks.append(Y_test[(Y_test > i*self.num_classes -1) & (Y_test < (i+1)*self.num_classes)]-i*self.num_classes)
  
            # Load CIFAR100 dataset
            X_train,Y_train=np.load('./datasets/train_cifar100.npz')['X'],np.load('./datasets/train_cifar100.npz')['Y']
            X_test,Y_test=np.load('./datasets/test_cifar100.npz')['X'],np.load('./datasets/test_cifar100.npz')['Y']
            X_train = torch.tensor(X_train, dtype=torch.float32)/255
            Y_train = torch.tensor(Y_train, dtype=torch.long)
            X_test = torch.tensor(X_test, dtype=torch.float32)/255
            Y_test = torch.tensor(Y_test, dtype=torch.long)
            num_task = 100//self.num_classes
            for i in range(num_task):
                X_train_tasks.append(X_train[(Y_train > i*self.num_classes -1) & (Y_train < (i+1)*self.num_classes)])
                X_test_tasks.append(X_test[(Y_test > i*self.num_classes -1) & (Y_test < (i+1)*self.num_classes)])
                Y_train_tasks.append(Y_train[(Y_train > i*self.num_classes -1) & (Y_train < (i+1)*self.num_classes)]-i*self.num_classes)
                Y_test_tasks.append(Y_test[(Y_test > i*self.num_classes -1) & (Y_test < (i+1)*self.num_classes)]-i*self.num_classes)
      
                
            self.X_train = X_train_tasks 
            self.Y_train = Y_train_tasks
            self.X_test = X_test_tasks
            self.Y_test = Y_test_tasks
            
        if self.dataset_name == 'CIFAR20':
            
            with open('./datasets/cifar-100-python/train', "rb") as f:
                data = pickle.load(f, encoding="bytes")
                X_train  = data[b"data"]
                Y_train_fine = np.array(data[b"fine_labels"])
                Y_train  = np.array(data[b"coarse_labels"])
                
            with open('./datasets/cifar-100-python/train', "rb") as f:
                data = pickle.load(f, encoding="bytes")
                X_test = data[b"data"]
                Y_test_fine = np.array(data[b"fine_labels"])
                Y_test = np.array(data[b"coarse_labels"])
                
            X_train = X_train.reshape((len(X_train),3,32,32))/255
            X_test = X_test.reshape((len(X_test),3,32,32))/255
            
            fine_to_coarse = {}
            for (fine, coarse) in zip(Y_test_fine, Y_test):
                if fine not in fine_to_coarse:
                    fine_to_coarse[fine] = coarse

            class_families = []
            for coarse in range(20):
                class_families.append([])
                for fine in range(100):
                    if fine_to_coarse[fine] == coarse:
                        class_families[coarse].append(fine)
                        
            class_families = np.array(class_families, dtype=int)           
            X_train_tasks = []
            Y_train_tasks = []
            X_test_tasks = []
            Y_test_tasks = []    
            # Generate random permutation for each family
            random_permutations = [np.random.permutation(family) for family in class_families]
            # Create 5 tasks
            for task in range(5):
               current_classes = [random_permutations[i][task] for i in range(20)]
               # Get the training and test samples for the current task's classes
               X_train_task = X_train[np.isin(Y_train_fine, current_classes)]
               Y_train_task = Y_train_fine[np.isin(Y_train_fine, current_classes)]
               X_test_task = X_test[np.isin(Y_test_fine, current_classes)]
               Y_test_task = Y_test_fine[np.isin(Y_test_fine, current_classes)]
              
               # Use np.vectorize to map fine labels to coarse labels
               map_to_coarse = np.vectorize(fine_to_coarse.get)
               Y_train_task_mapped = map_to_coarse(Y_train_task)  # Apply mapping
               Y_test_task_mapped = map_to_coarse(Y_test_task)    # Apply mapping
               # Append the task data
               X_train_tasks.append(torch.tensor(X_train_task, dtype=torch.float32))
               Y_train_tasks.append(torch.tensor(Y_train_task_mapped, dtype=torch.long))
               X_test_tasks.append(torch.tensor(X_test_task, dtype=torch.float32))
               Y_test_tasks.append(torch.tensor(Y_test_task_mapped, dtype=torch.long))
               
            self.X_train = X_train_tasks 
            self.Y_train = Y_train_tasks
            self.X_test = X_test_tasks
            self.Y_test = Y_test_tasks
            
        if self.dataset_name == 'KMNIST':
            # Load KMNIST dataset 
            X_train,Y_train=np.load('./datasets/k49-train-imgs.npz')['arr_0'],np.load('./datasets/k49-train-labels.npz')['arr_0']
            X_test,Y_test=np.load('./datasets/k49-test-imgs.npz')['arr_0'],np.load('./datasets/k49-test-labels.npz')['arr_0']
            X_test = torch.tensor(X_test, dtype=torch.float32)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            self.X_train = torch.flatten(X_train, 1)/255
            self.X_test = torch.flatten(X_test, 1)/255
            self.Y_test = torch.tensor(Y_test, dtype=torch.long)
            self.Y_train = torch.tensor(Y_train, dtype=torch.long)

        if self.dataset_name == 'ANIMALS':
           # Load CIFAR10 dataset
           X_train,Y_train=np.load('./datasets/domain_inc_animals_train_4.npz')['X'],np.load('./datasets/domain_inc_animals_train_4.npz')['Y']
           X_test,Y_test=np.load('./datasets/domain_inc_animals_test_4.npz')['X'],np.load('./datasets/domain_inc_animals_test_4.npz')['Y']
           X_train_tasks = []
           Y_train_tasks = []
           X_test_tasks = []
           Y_test_tasks = []
           X_train=X_train[:,:self.archi_fcnn[0]]
           X_test=X_test[:,:self.archi_fcnn[0]]
           class_families = [
               [0, 5, 10, 15],   # Family 0
               [1, 6, 11, 16],   # Family 1
               [2, 7, 12, 17],   # Family 2
               [3, 8, 13, 18],   # Family 3
               [4, 9, 14, 19]    # Family 4
            ]
           
          
           random_permutations = [np.random.permutation(family) for family in class_families]
            
            # Function to balance classes by oversampling
           def balance_classes(X, Y):
                unique_classes, counts = np.unique(Y, return_counts=True)
                # max_count = np.max(counts)
                max_count = 720 #Classe that has the more images
                
                X_balanced, Y_balanced = [], []
                for cls in unique_classes:
                    cls_indices = np.where(Y == cls)[0]
                    cls_samples = X[cls_indices]
                    num_repeats = max_count // counts[cls]  # full duplications needed
                    num_remaining = max_count % counts[cls]  # extra images to reach max count
        
                    X_balanced.append(np.tile(cls_samples, (num_repeats, 1)))
                    X_balanced.append(cls_samples[:num_remaining])
        
                    Y_balanced.append(np.full(max_count, cls))
        
                return np.concatenate(X_balanced), np.concatenate(Y_balanced)
        
            # Create 5 tasks and balance each task
           for task in range(4):
                current_classes = [random_permutations[i][task] for i in range(5)]
                
                # Get the training and test samples for the current task's classes
                X_train_task = X_train[np.isin(Y_train, current_classes)]
                Y_train_task = Y_train[np.isin(Y_train, current_classes)]
                X_test_task = X_test[np.isin(Y_test, current_classes)]
                Y_test_task = Y_test[np.isin(Y_test, current_classes)]
                
                # Remap the class labels to be between 0 and 4
                Y_train_task_mapped = Y_train_task % 5
                Y_test_task_mapped = Y_test_task % 5
                
                # Balance the training set for this task
                X_train_balanced, Y_train_balanced = balance_classes(X_train_task, Y_train_task_mapped)
                
                # Append the task data
                X_train_tasks.append(torch.tensor(X_train_balanced, dtype=torch.float32))
                Y_train_tasks.append(torch.tensor(Y_train_balanced, dtype=torch.long))
                X_test_tasks.append(torch.tensor(X_test_task, dtype=torch.float32))
                Y_test_tasks.append(torch.tensor(Y_test_task_mapped, dtype=torch.long))
                
            # Save the balanced datasets
           self.X_train = X_train_tasks
           self.Y_train = Y_train_tasks
           self.X_test = X_test_tasks
           self.Y_test = Y_test_tasks
           
    def load_ood(self,):
        if self.dataset_name == 'CIFAR10':
            # Load CIFAR100 test dataset for uncertainty measure
            X_test,Y_test=np.load('./datasets/test_cifar100.npz')['X'],np.load('./datasets/test_cifar100.npz')['Y']
            self.X_ood = torch.tensor(X_test, dtype=torch.float32)/255
            self.Y_odd = torch.tensor(Y_test, dtype=torch.long)
        
        if self.dataset_name == 'MNIST':
            # Load KMNIST 10k test dataset for uncertainty measure 
            X_test,Y_test=np.load('./datasets/k49-test-imgs.npz')['arr_0'],np.load('./datasets/k49-test-labels.npz')['arr_0']
            X_test = torch.tensor(X_test[:10000,], dtype=torch.float32)/255
            self.X_ood = torch.flatten(X_test, 1)
            self.Y_ood = torch.tensor(Y_test[:10000,], dtype=torch.long)
            permutation=np.random.permutation(784)
            # self.X_ood_P = X_test[:,permutation]
        if self.dataset_name == 'ANIMALS':
            # Load CIFAR10 dataset
            X_ood,Y_ood=np.load('./datasets/domain_inc_animals_ood.npz')['X'],np.load('./datasets/domain_inc_animals_ood.npz')['Y']
            #We hide the class sypder
          
            self.X_ood = torch.tensor(X_ood, dtype=torch.float32)[:,:self.archi_fcnn[0]]
    
    def normalize(self,):
        self.min = self.X_train.min()
        self.max = self.X_train.max()
        self.X_train =(self.X_train - self.min)/(self.max-self.min)
        self.X_test = (self.X_test - self.min)/(self.max-self.min)
        if self.X_ood is not None:
            self.X_ood = (self.X_ood - self.min)/(self.max-self.min)
        if self.X_ood_P is not None:
            self.X_ood_P = (self.X_ood_P - self.min)/(self.max-self.min)
            
    def discretize(self,num):
        self.standardize()
        disc = torch.linspace(0,1, num)
        self.X_train = disc[torch.argmin((self.X_train[None,:,:]-disc[:,None,None])**2, axis=0)]
        self.X_test = disc[torch.argmin((self.X_test[None,:,:]-disc[:,None,None])**2, axis=0)]
        if self.X_ood is not None:
            self.X_ood = disc[torch.argmin((self.X_ood[None,:,:]-disc[:,None,None])**2, axis=0)]
        if self.X_ood_P is not None:
            self.X_ood_P = disc[torch.argmin((self.X_ood_P[None,:,:]-disc[:,None,None])**2, axis=0)]
    
    def shift_and_scale(self,shift,scale):
        self.X_train = self.X_train*scale +shift
        self.X_test = self.X_test*scale +shift
        if self.X_ood is not None:
            self.X_ood = self.X_ood*scale +shift
        if self.X_ood_P is not None:
            self.X_ood = self.X_ood*scale +shift
        
    def standardize(self,):
        self.mean = self.X_train.mean()
        self.std = self.X_train.std()
        self.X_train =(self.X_train - self.mean)/self.std
        self.X_test = (self.X_test - self.mean)/self.std
        if self.X_ood is not None:
            self.X_ood = (self.X_ood - self.mean)/self.std
            
            
    def imshow(self,img):
        if len(img.size())!=3:
            s = int(img.size()[-1]**0.5)
            img=img.reshape(s,s,1)
            npimg = img.numpy()
            plt.imshow(npimg,cmap='Greys_r')
        else:
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
            

