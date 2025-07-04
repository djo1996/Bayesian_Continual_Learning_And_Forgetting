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
This module implements standard (deterministic) models used for experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import Module
from torch.autograd import Function

# Check if CUDA is available
if torch.cuda.is_available():
    # Set the default data type for tensors
    torch.set_default_dtype(torch.float32)
    # Set the default device to CUDA (GPU)
    torch.set_default_device('cuda')
    # Print a message indicating that CUDA is available


class CNN_CIFAR_K_HEAD_DET(nn.Module):
    """
    Convolutional neural network used for task incremental learning on Split CIFAR10. 
    Same architecture as in F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
    """
    def __init__(self, args_dict):
        super(CNN_CIFAR_K_HEAD_DET, self).__init__()
        self.coeff_likeli = args_dict['coeff_likeli']
        self.reduction = args_dict['reduction']
        self.num_classes = args_dict['num_classes']
        self.num_heads = args_dict['num_heads']
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout(p=0.25)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2304, 512)
        self.heads = nn.ModuleList()
        for k in range(self.num_heads):
            self.heads.append(nn.Linear(512, self.num_classes))
        self.transform_train = transforms.RandomHorizontalFlip(0.) #NOT USED AT THE END
        
        
    def forward(self, x, head=0):
        """ 
        Forward pass through the network. 
        When samples=0, the model is deterministic, but we need samples_dim=1 for reshaping to work.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = F.relu(x)
        x = self.drop_conv(self.pool(x))

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)

        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        

        x = torch.flatten(x, 1)  # Flatten feature maps based on pooled dimensions
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_fc(x)
        
        
        
        # choose the head for split CIFAR, task incremental learning scenario with known task boundaries
        x = self.heads[head](x)
        
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, x, target, head=1):
        """
        Compute the Negative Log Likelihood
        """
        outputs = self.forward(self.transform_train(x), head=head)
        negative_log_likelihood = F.nll_loss(outputs, target, reduction=self.reduction)
        return negative_log_likelihood*self.coeff_likeli
    
    

    def conv_parameters(self, recurse: bool = True):
        """
        Yield parameters associated with convolutions.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.startswith('conv'):
                yield param

    def fc_parameters(self, recurse: bool = True):
        """
        Yield parameters associated with fully connected layers.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.startswith('fc'):
                yield param

   

class CNN_CIFAR_K_HEAD_EWC(nn.Module):
    """
    Convolutional neural network used for task incremental learning on Split CIFAR10. 
    Same architecture as in F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
    """
    def __init__(self, args_dict):
        super(CNN_CIFAR_K_HEAD_EWC, self).__init__()
        self.coeff_likeli = args_dict['coeff_likeli']
        self.reduction = args_dict['reduction']
        self.num_classes = args_dict['num_classes']
        self.num_heads = args_dict['num_heads']
        self.Lambda = args_dict['lambda']
        self.batch_size_fisher = args_dict['batch_size_fisher']
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout(p=0.25)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2304, 512)
        self.heads = nn.ModuleList()
        for k in range(self.num_heads):
            self.heads.append(nn.Linear(512, self.num_classes))
        self.transform_train = transforms.RandomHorizontalFlip(0.)
        self.task_seen=0
        self.fisher_param={}
        self.mean_param={}

    def new_consolidation(self,X,Y,head):
        self.create_ewc_params(self.task_seen)
        self.compute_fisher(X,Y,head,self.task_seen)
        self.task_seen+=1

    def create_ewc_params(self,task):
        with torch.no_grad():
            for i, (name, param) in enumerate(self.named_parameters(recurse=True)):
                self.fisher_param[name+"task"+str(task)]=torch.zeros_like(param.data)
                self.mean_param[name+"task"+str(task)] = param.data.clone()
    
    def compute_fisher(self,X,Y,head,task):
        num_batches=len(X)//self.batch_size_fisher
        self.train()
        for batch_idx in range(num_batches):
            Xb = X[batch_idx * self.batch_size_fisher:(batch_idx + 1) * self.batch_size_fisher]
            Yb = Y[batch_idx * self.batch_size_fisher:(batch_idx + 1) * self.batch_size_fisher]
            self.zero_grad()
            loss_nll = self.loss_nll(Xb, Yb, head=head)
            loss_nll.backward()
            for i, (name, param) in enumerate(self.named_parameters(recurse=True)):
                if param.grad is not None:
                    self.fisher_param[name+"task"+str(task)]+=((param.grad.data)**2)/num_batches
            
    def ewc_loss(self,task):
        ewc_loss = 0
        for i, (name, param) in enumerate(self.named_parameters(recurse=True)):
                for j in range(task):
                    ewc_loss += (0.5*self.Lambda*self.fisher_param[name+"task"+str(j)]*(self.mean_param[name+"task"+str(j)] - param)**2).sum()
        return ewc_loss

    def forward(self, x, head=0):
        """ 
        Forward pass through the network. 
        When samples=0, the model is deterministic, but we need samples_dim=1 for reshaping to work.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = F.relu(x)
        x = self.drop_conv(self.pool(x))

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)

        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        

        x = torch.flatten(x, 1)  # Flatten feature maps based on pooled dimensions
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_fc(x)
        
        
        
        # choose the head for split CIFAR, task incremental learning scenario with known task boundaries
        x = self.heads[head](x)
        
        x = F.log_softmax(x, dim=1)
        return x

    def loss_nll(self, x, target, head=1):
        """
        Compute the Negative Log Likelihood
        """
        outputs = self.forward(self.transform_train(x), head=head)
        #Important that it is mean for our implementation of EWC
        negative_log_likelihood = F.nll_loss(outputs, target, reduction="mean")

        return negative_log_likelihood
    
    
    def loss(self, x, target, head=1):
        """
        Compute the Negative Log Likelihood
        """
        outputs = self.forward(self.transform_train(x), head=head)
        negative_log_likelihood = F.nll_loss(outputs, target, reduction="mean")
        ewc_loss = self.ewc_loss(self.task_seen)
        return negative_log_likelihood + ewc_loss


class CNN_CIFAR_K_HEAD_SI(nn.Module):
    """
    Convolutional neural network used for task incremental learning on Split CIFAR10. 
    Same architecture as in F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
    """
    def __init__(self, args_dict):
        super(CNN_CIFAR_K_HEAD_SI, self).__init__()
        self.coeff_likeli = args_dict['coeff_likeli']
        self.reduction = args_dict['reduction']
        self.num_classes = args_dict['num_classes']
        self.num_heads = args_dict['num_heads']
        self.batch_size_fisher = args_dict['batch_size_fisher']
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout(p=0.25)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2304, 512)
        self.heads = nn.ModuleList()
        for k in range(self.num_heads):
            self.heads.append(nn.Linear(512, self.num_classes))
        self.transform_train = transforms.RandomHorizontalFlip(0.)
        self.task_seen=0
      
        #For Synaptic Intelligence
        self.damping_factor = args_dict['damping_factor']
        self.c_si = args_dict['c_si']
        self.importance = {}
        self.mean_param = {}
        self.w = {}
        for i, (name, param) in enumerate(self.named_parameters(recurse=True)):
            self.w[name] = param.clone().detach().zero_()



    def create_si_params(self,):
        task = self.task_seen
        with torch.no_grad():
            for i, (name, param) in enumerate(self.named_parameters(recurse=True)):
                self.importance[name+"task"+str(task)]=torch.zeros_like(param.data)
                self.mean_param[name+"task"+str(task)] = param.data.clone().detach() #first used to remeber old param to compute importance, then updated and used to define SI loss
    
    # Below are some custom functions that can be used for specific experiments
    def calculate_importance(self,):
        """
        function taken from https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/agents/regularization.py
        """
        task = self.task_seen
        self.task_seen += 1
        for i, (name, param) in enumerate(self.named_parameters(recurse=True)):
            delta_theta = param.detach().data - self.mean_param[name+"task"+str(task)]
            self.importance[name+"task"+str(task)] += torch.relu(self.w[name] / (delta_theta**2 + self.damping_factor)) # Relu just in case
            self.mean_param[name+"task"+str(task)] = param.data.clone().detach() #update mean param for si loss
            self.w[name].zero_()  

    def si_loss(self,task):
        si_loss = 0
        for i, (name, param) in enumerate(self.named_parameters(recurse=True)):
                for j in range(task):
                    si_loss += (self.c_si*self.importance[name+"task"+str(j)]*(self.mean_param[name+"task"+str(j)] - param)**2).sum()
        return si_loss

    def forward(self, x, head=0):
        """ 
        Forward pass through the network. 
        When samples=0, the model is deterministic, but we need samples_dim=1 for reshaping to work.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = F.relu(x)
        x = self.drop_conv(self.pool(x))

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)

        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        

        x = torch.flatten(x, 1)  # Flatten feature maps based on pooled dimensions
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_fc(x)
        
        
        
        # choose the head for split CIFAR, task incremental learning scenario with known task boundaries
        x = self.heads[head](x)
        
        x = F.log_softmax(x, dim=1)
        return x

    def loss_nll(self, x, target, head=1):
        """
        Compute the Negative Log Likelihood
        """
        outputs = self.forward(self.transform_train(x), head=head)
        #Important that it is mean for our implementation of EWC
        negative_log_likelihood = F.nll_loss(outputs, target, reduction="mean")

        return negative_log_likelihood
    
    
    def loss(self, x, target, head=1):
        """
        Compute the Negative Log Likelihood
        """
        outputs = self.forward(self.transform_train(x), head=head)
        negative_log_likelihood = F.nll_loss(outputs, target, reduction="mean")
        si_loss = self.si_loss(self.task_seen)
        return negative_log_likelihood + si_loss
    

class CNN_CIFAR_DET(nn.Module):
    """
    Convolutional neural network used for classic learning on CIFAR10. 
    Same architecture as in F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
    """
    def __init__(self, args_dict):
        super(CNN_CIFAR_DET, self).__init__()
        self.coeff_likeli=args_dict['coeff_likeli']
        self.reduction=args_dict['reduction']
        self.num_classes = args_dict['num_classes']
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout(p=0.25)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.transform_train = transforms.RandomHorizontalFlip(0.)


    def forward(self, x, head=1):
        """ 
        Forward pass through the network. 
        If in training mode we apply a random horizontal flip to the input
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        
     
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        
       
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        
       
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        
        x = torch.flatten(x, 1)  # Flatten feature maps based on pooled dimensions

        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, x, target):
        """
        Compute the Negative Log Likelihood.
        """
        outputs = self.forward(self.transform_train(x))
        negative_log_likelihood = F.nll_loss(outputs, target, reduction=self.reduction)
        return negative_log_likelihood*self.coeff_likeli 
    

      
   
class FCNN_DET(nn.Module):
    """
    Fully connected neural network used for classic and domain incremental learning on MNIST.
    The activation function "elephant" is taken from: Q. Lan "Elephant Neural Networks: Born to Be a Continual Learner" (2023).
    It provides better results for domain incremental learning.
    This model is also used to measure the quality of synaptic importance in EWC/SI/MESU.
    """
    def __init__(self, args_dict):
        """
        Initialize the Bayesian Neural Network.

        Parameters:
        - archi_fcnn: A list where the first element is num_in, the last is num_out, and intermediate values are hidden layer sizes.
        - sigma_init: Initial sigma for the weights.
        - sigma_prior: Prior sigma for the weights.
        - a, d: Parameters for the elephant activation function.
        - activation: Type of activation function to use.
        """
        super(FCNN_DET, self).__init__()
        
        archi_fcnn=args_dict['archi_fcnn']
        a,d=args_dict['elephant_params']
        self.num_classes = args_dict['num_classes']
        activation=args_dict['activation']
        self.coeff_likeli=args_dict['coeff_likeli']
        self.num_out = archi_fcnn[-1]
        self.reduction=args_dict['reduction']
        self.layers = nn.ModuleList()
        for i in range(len(archi_fcnn) - 1):
            self.layers.append(nn.Linear(archi_fcnn[i], archi_fcnn[i + 1]))
        valid_activations = {'Tanh', 'Relu',  'Hardtanh'}
        if activation not in valid_activations:
            raise ValueError("Invalid activation name {!r}, should be one of {}".format(activation, valid_activations))
        if activation == 'Tanh':
            self.act = torch.nn.Tanh()
        elif activation == 'Hardtanh':
            self.act = torch.nn.Hardtanh(min_val=-0.1, max_val=0.1)
        elif activation == 'Relu':
            self.act = torch.nn.ReLU()

            
        #For Synaptic Intelligence
        self.damping_factor = 0.1
        self.w = {}
        for idx, p in enumerate(self.parameters()):
            self.w[idx] = p.clone().detach().zero_()
        self.initial_params = {}
        for idx, p in enumerate(self.parameters()):
            self.initial_params[idx] = p.clone().detach()

    def forward(self, x):
        """
        Forward propagation through the network.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)
        pred = F.log_softmax(x, dim=1)
        return pred
    
    def loss(self, x, target):
        """
        Compute the Negative Log Likelihood.
        """
        outputs = self.forward(x)
        negative_log_likelihood = F.nll_loss(outputs, target, reduction=self.reduction)
        return negative_log_likelihood*self.coeff_likeli
   
    # Below are some custom functions that can be used for specific experiments
    def calculate_importance(self,):
        """
        function taken from https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/agents/regularization.py
        """

        importance = {}
        for idx, p in enumerate(self.parameters()):
            importance[idx] = p.clone().detach().fill_(0)  # zero initialized
        prev_params = self.initial_params
    
        # Calculate or accumulate the Omega (the importance matrix)
        for idx, p in importance.items():
            delta_theta = list(self.parameters())[idx].detach() - prev_params[idx]
            p += self.w[idx] / (delta_theta**2 + self.damping_factor)
            self.w[idx].zero_()
        #return only the impportance of the last layer weight
        return importance[2].cpu().detach().numpy().flatten()
    
    