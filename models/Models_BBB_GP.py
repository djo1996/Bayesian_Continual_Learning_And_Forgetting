
#!/usr/bin/env python3                     
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT AND BSD-3-Clause 
#
# Code for “Bayesian continual learning and forgetting in neural networks”
# (arXiv:2504.13569)
# Portions adapted from the PyTorch project (BSD-3-Clause) 
#
# Author: Djohan Bonnet  <djohan.bonnet@gmail.com>
# Date: 2025-04-18
"""
This module implements Bayes By Backprop (with Gaussian Prior) models used for experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import Conv2d_BBB_GP, Linear_BBB_GP

# NOT USED IN THE MAIN ARTICLE, THIS IS TO IMPLEMENT BAYES BY BACKPROP
# I LEAVE IT HERE IN CASE SOMEONE IS INTERESTED


# Check if CUDA is available
if torch.cuda.is_available():
    # Set the default data type for tensors
    torch.set_default_dtype(torch.float32)
    # Set the default device to CUDA (GPU)
    torch.set_default_device('cuda')
    # Print a message indicating that CUDA is available


    
class CNN_CIFAR_K_HEAD_BBB_GP(nn.Module):
    """
    Convolutional neural network used for task incremental learning on Split CIFAR10. 
    Same architecture as in F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
    """
    def __init__(self, args_dict):
        super(CNN_CIFAR_K_HEAD_BBB_GP, self).__init__()
        sigma_init_fc=args_dict['sigma_init']
        sigma_init_conv=args_dict['sigma_init_conv']
        sigma_prior=args_dict['sigma_prior']
        self.coeff_likeli=args_dict['coeff_likeli']
        self.coeff_kl =args_dict['coeff_kl']
        self.reduction = args_dict['reduction']
        self.num_classes = args_dict['num_classes']
        self.num_heads = args_dict['num_heads']
        self.num_batches = None #Value will be attributed during training
        
        self.conv1 = Conv2d_BBB_GP(3, 32, 3, padding='same', sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv2 = Conv2d_BBB_GP(32, 32, 3, sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv3 = Conv2d_BBB_GP(32, 64, 3, padding='same', sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv4 = Conv2d_BBB_GP(64, 64, 3, sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout(p=0.25)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc1 = Linear_BBB_GP(2304, 512, sigma_init=sigma_init_fc, sigma_prior=sigma_prior)
        self.heads=[]
        for k in range(self.num_heads):
            self.heads.append(Linear_BBB_GP(512, self.num_classes, sigma_init=sigma_init_fc, sigma_prior=sigma_prior))
        self.transform_train = transforms.RandomHorizontalFlip(0.5)
        
    def forward(self, x, samples=1, head=1):
        """ 
        Forward pass through the network. 
        When samples=0, the model is deterministic, but we need samples_dim=1 for reshaping to work.
        """
        samples_dim = max(samples, 1)
        x = x.repeat(samples_dim,1,1,1) 
        x = self.conv1(x, samples)
        x = F.relu(x)
        x = self.conv2(x, samples)
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        x = self.conv3(x, samples)
        x = F.relu(x)
        x = self.conv4(x, samples)
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        
        # Reshape across sampling and batch dimensions after pooling
        x = x.view(samples_dim, -1, x.size(1), x.size(2), x.size(3))
        x = torch.flatten(x, 2)  # Flatten feature maps based on pooled dimensions
        x = self.fc1(x, samples)
        x = F.relu(x)
        x = self.drop_fc(x)  
        
        # choose the head for split CIFAR, task incremental learning scenario with known task boundaries
        x = self.heads[head](x, samples)
        
        
        x = F.log_softmax(x, dim=2)
        return x

    def loss(self, x, target, samples=1, head=1):
        """
        Compute the Negative Log Likelihood estimated by Monte Carlo sampling and the kl divergence in close form.
        """
        outputs = self.forward(self.transform_train(x), samples=samples, head=head)
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction=self.reduction)
        kl = self.kl(head) / self.num_batches
        return negative_log_likelihood *self.coeff_likeli+ kl*self.coeff_kl
    
    def kl(self,head):
        """
        Compute the KL divergence between the variational distribution and the prior.
        - The coefficient should be 1 in the strict bayesian framework
        - However for cifar we must have coeff<<1 so that we can learn. 
        - Otherwise the regularization is too strong.This is not uncommon in variational methods"
        """
        kl = self.conv1.kl() + self.conv2.kl() + self.conv3.kl() + self.conv4.kl() + self.fc1.kl() 
        if head ==1:
            kl += self.head1.kl()
        elif head ==2:
            kl += self.head2.kl()
        return kl

    # Below are some custom functions that can be used for specific experiments

    def Mean_parameters(self, recurse: bool = True):
        """
        Yield mean parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('mu'):
                yield param

    def Std_parameters(self, recurse: bool = True):
        """
        Yield standard deviation parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('sigma'):
                yield param

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

    def Bayesian_parameters(self, recurse: bool = True):
        """
        Yield parameters associated with Bayesian inference (mu and sigma).
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('sigma') or name.endswith('mu'):
                yield param

    def Deterministic_parameters(self, recurse: bool = True):
        """
        Yield parameters not associated with Bayesian inference (i.e., non-mu and non-sigma parameters).
        """
        for name, param in self.named_parameters(recurse=recurse):
            if not (name.endswith('sigma') or name.endswith('mu')):
                yield param


class CNN_CIFAR_K_HEAD_MIXTE_BBB_GP(nn.Module):
    """
    Convolutional neural network used for task incremental learning on Split CIFAR10. 
    Same architecture as in F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
    """
    def __init__(self, args_dict):
        super(CNN_CIFAR_K_HEAD_MIXTE_BBB_GP, self).__init__()
        sigma_init_fc=args_dict['sigma_init']
        sigma_init_conv=args_dict['sigma_init_conv']
        sigma_prior=args_dict['sigma_prior']
        self.coeff_likeli=args_dict['coeff_likeli']
        self.coeff_kl =args_dict['coeff_kl']
        self.reduction = args_dict['reduction']
        self.num_classes = args_dict['num_classes']
        self.num_heads = args_dict['num_heads']
        self.num_batches = None #Value will be attributed during training
        
        self.conv1 = Conv2d_BBB_GP(3, 32, 3, padding='same', sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv2 = Conv2d_BBB_GP(32, 32, 3, sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv3 = Conv2d_BBB_GP(32, 64, 3, padding='same', sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv4 = Conv2d_BBB_GP(64, 64, 3, sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout(p=0.25)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc1 = Linear_BBB_GP(2304, 512, sigma_init=sigma_init_fc, sigma_prior=sigma_prior)
        self.heads=[]
        for k in range(self.num_heads):
            self.heads.append(Linear_BBB_GP(512, self.num_classes, sigma_init=sigma_init_fc, sigma_prior=sigma_prior))
        self.transform_train = transforms.RandomHorizontalFlip(0.5)
        
    def forward(self, x, samples=1, head=1):
        """ 
        Forward pass through the network. 
        When samples=0, the model is deterministic, but we need samples_dim=1 for reshaping to work.
        """

        x = self.conv1(x, 0)
        x = F.relu(x)
        x = self.conv2(x, 0)
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        x = self.conv3(x, 0)
        x = F.relu(x)
        x = self.conv4(x, 0)
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        
        x = torch.flatten(x, 1)  # Flatten feature maps based on pooled dimensions
        x = self.fc1(x, samples)
        x = F.relu(x)
        x = self.drop_fc(x)  
        
        # choose the head for split CIFAR, task incremental learning scenario with known task boundaries
        x = self.heads[head](x, samples)
        
        
        x = F.log_softmax(x, dim=2)
        return x

    def loss(self, x, target, samples=1, head=1):
        """
        Compute the Negative Log Likelihood estimated by Monte Carlo sampling and the kl divergence in close form.
        """
        outputs = self.forward(self.transform_train(x), samples=samples, head=head)
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction=self.reduction)
        kl = self.kl(head) / self.num_batches
        return negative_log_likelihood *self.coeff_likeli+ kl*self.coeff_kl
    
    def kl(self,head):
        """
        Compute the KL divergence between the variational distribution and the prior.
        - The coefficient should be 1 in the strict bayesian framework
        - However for cifar we must have coeff<<1 so that we can learn. 
        - Otherwise the regularization is too strong.This is not uncommon in variational methods"
        """
        kl = self.conv1.kl() + self.conv2.kl() + self.conv3.kl() + self.conv4.kl() + self.fc1.kl() 
        if head ==1:
            kl += self.head1.kl()
        elif head ==2:
            kl += self.head2.kl()
        return kl

    # Below are some custom functions that can be used for specific experiments

    def Mean_parameters(self, recurse: bool = True):
        """
        Yield mean parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('mu'):
                yield param

    def Std_parameters(self, recurse: bool = True):
        """
        Yield standard deviation parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('sigma'):
                yield param

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

    def Bayesian_parameters(self, recurse: bool = True):
        """
        Yield parameters associated with Bayesian inference (mu and sigma).
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('sigma') or name.endswith('mu'):
                yield param

    def Deterministic_parameters(self, recurse: bool = True):
        """
        Yield parameters not associated with Bayesian inference (i.e., non-mu and non-sigma parameters).
        """
        for name, param in self.named_parameters(recurse=recurse):
            if not (name.endswith('sigma') or name.endswith('mu')):
                yield param
class CNN_CIFAR_BBB_GP(nn.Module):
    """
    Convolutional neural network used for classic learning on CIFAR10. 
    Same architecture as in F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
    """
    def __init__(self, args_dict):
        super(CNN_CIFAR_BBB_GP, self).__init__()
        
        sigma_init_fc=args_dict['sigma_init']
        sigma_init_conv=args_dict['sigma_init_conv']
        sigma_prior=args_dict['sigma_prior']
        self.coeff_likeli=args_dict['coeff_likeli']
        self.coeff_kl =args_dict['coeff_kl']
        self.reduction=args_dict['reduction']
        self.num_classes = args_dict['num_classes']
        self.num_batches = None #Value will be attributed during training
        
        self.conv1 = Conv2d_BBB_GP(3, 32, 3, padding='same', sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv2 = Conv2d_BBB_GP(32, 32, 3, sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv3 = Conv2d_BBB_GP(32, 64, 3, padding='same', sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv4 = Conv2d_BBB_GP(64, 64, 3, sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout(p=0.25)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc1 = Linear_BBB_GP(2304, 512, sigma_init=sigma_init_fc, sigma_prior=sigma_prior)
        self.fc2 = Linear_BBB_GP(512, self.num_calsses, sigma_init=sigma_init_fc, sigma_prior=sigma_prior)
        self.transform_train = transforms.RandomHorizontalFlip(0.5)
        #parameters for BBB 
    def forward(self, x, samples=1):
        """ 
        Forward pass through the network. 
        If in training mode we apply a random horizontal flip to the input
        When samples=0, the model is deterministic, but we need samples_dim=1 for reshaping to work.
        """
        samples_dim = max(samples, 1)
        x = x.repeat(samples_dim,1,1,1) 
        x = self.conv1(x, samples)
        x = F.relu(x)
        x = self.conv2(x, samples)
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        x = self.conv3(x, samples)
        x = F.relu(x)
        x = self.conv4(x, samples)
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        
        # Reshape across sampling and batch dimensions after pooling
        x = x.view(samples_dim, -1, x.size(1), x.size(2), x.size(3))
        x = torch.flatten(x, 2)  # Flatten feature maps based on pooled dimensions
        x = self.fc1(x, samples)
        x = F.relu(x)
        x = self.drop_fc(x)
        x = self.fc2(x, samples)
        
        x = F.log_softmax(x, dim=2)
        return x

    def loss(self, x, target, samples=1 ):
        """
        Compute the Negative Log Likelihood estimated by Monte Carlo sampling and the kl divergence in close form.
        """
        outputs = self.forward(self.transform_train(x), samples=samples)
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction=self.reduction)
        kl = self.kl() / self.num_batches
        return negative_log_likelihood*self.coeff_likeli + kl*self.coeff_kl
    
    def kl(self,):
        """
        Compute the KL divergence between the variational distribution and the prior.
        - The coefficient should be 1 in the strict bayesian framework
        - However for cifar we must have coeff<<1 so that we can learn. 
        - Otherwise the regularization is too strong.This is not uncommon in variational methods"
        """
        kl = self.conv1.kl() + self.conv2.kl() + self.conv3.kl() + self.conv4.kl() + self.fc1.kl() + self.fc2.kl() 
        return kl
    
    # Below are some custom functions that can be used for specific experiments
    def Mean_parameters(self, recurse: bool = True):
        """
        Yield mean parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('mu'):
                yield param
    
    def Std_parameters(self, recurse: bool = True):
        """
        Yield standard deviation parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('sigma'):
                yield param
                
class CNN_CIFAR_MIXTE_BBB_GP(nn.Module):
    """
    Convolutional neural network used for classic learning on CIFAR10. 
    Same architecture as in F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
    """
    def __init__(self, args_dict):
        super(CNN_CIFAR_MIXTE_BBB_GP, self).__init__()
        
        sigma_init_fc=args_dict['sigma_init']
        sigma_init_conv=args_dict['sigma_init_conv']
        sigma_prior=args_dict['sigma_prior']
        self.coeff_likeli=args_dict['coeff_likeli']
        self.coeff_kl =args_dict['coeff_kl']
        self.reduction=args_dict['reduction']
        self.num_classes = args_dict['num_classes']
        self.num_batches = None #Value will be attributed during training
        
        self.conv1 = Conv2d_BBB_GP(3, 32, 3, padding='same', sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv2 = Conv2d_BBB_GP(32, 32, 3, sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv3 = Conv2d_BBB_GP(32, 64, 3, padding='same', sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.conv4 = Conv2d_BBB_GP(64, 64, 3, sigma_init=sigma_init_conv, sigma_prior=sigma_prior)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout(p=0.25)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc1 = Linear_BBB_GP(2304, 512, sigma_init=sigma_init_fc, sigma_prior=sigma_prior)
        self.fc2 = Linear_BBB_GP(512, self.num_classes, sigma_init=sigma_init_fc, sigma_prior=sigma_prior)
        self.transform_train = transforms.RandomHorizontalFlip(0.5)
        #parameters for BBB 
    def forward(self, x, samples=1):
        """ 
        Forward pass through the network. 
        If in training mode we apply a random horizontal flip to the input
        When samples=0, the model is deterministic, but we need samples_dim=1 for reshaping to work.
        """
        x = self.conv1(x, 0)
        x = F.relu(x)
        x = self.conv2(x, 0)
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        x = self.conv3(x, 0)
        x = F.relu(x)
        x = self.conv4(x, 0)
        x = F.relu(x)
        x = self.drop_conv(self.pool(x))
        
        # Reshape across sampling and batch dimensions after pooling
        x = torch.flatten(x, 1)  # Flatten feature maps based on pooled dimensions
        x = self.fc1(x, samples)
        x = F.relu(x)
        x = self.drop_fc(x)
        x = self.fc2(x, samples)
        
        x = F.log_softmax(x, dim=2)
        return x

    def loss(self, x, target, samples=1 ):
        """
        Compute the Negative Log Likelihood estimated by Monte Carlo sampling and the kl divergence in close form.
        """
        outputs = self.forward(self.transform_train(x), samples=samples)
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction=self.reduction)
        kl = self.kl() / self.num_batches
        return negative_log_likelihood*self.coeff_likeli + kl*self.coeff_kl
    
    def kl(self,):
        """
        Compute the KL divergence between the variational distribution and the prior.
        - The coefficient should be 1 in the strict bayesian framework
        - However for cifar we must have coeff<<1 so that we can learn. 
        - Otherwise the regularization is too strong.This is not uncommon in variational methods"
        """
        kl = self.conv1.kl() + self.conv2.kl() + self.conv3.kl() + self.conv4.kl() + self.fc1.kl() + self.fc2.kl() 
        return kl
    
    # Below are some custom functions that can be used for specific experiments
    def Mean_parameters(self, recurse: bool = True):
        """
        Yield mean parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('mu'):
                yield param
    
    def Std_parameters(self, recurse: bool = True):
        """
        Yield standard deviation parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('sigma'):
                yield param

class FCNN_BBB_GP(nn.Module):
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
        super(FCNN_BBB_GP, self).__init__()
        
        archi_fcnn=args_dict['archi_fcnn']
        sigma_init=args_dict['sigma_init']
        sigma_prior=args_dict['sigma_prior']
        mu_prior=args_dict['mu_prior']
        a,d=args_dict['elephant_params']
        activation=args_dict['activation']
        self.num_classes = archi_fcnn[-1]
        self.coeff_likeli = args_dict['coeff_likeli']
        self.coeff_kl = args_dict['coeff_kl']
        self.reduction = args_dict['reduction']
        self.num_batches = None #Value will be attributed during training

        self.layers = nn.ModuleList()
        for i in range(len(archi_fcnn) - 1):
            self.layers.append(Linear_BBB_GP(archi_fcnn[i], archi_fcnn[i + 1], sigma_init=sigma_init, sigma_prior=sigma_prior, mu_prior=mu_prior))



        valid_activations = {'Tanh', 'Relu', 'Hardtanh'}
        if activation not in valid_activations:
            raise ValueError("Invalid activation name {!r}, should be one of {}".format(activation, valid_activations))

        if activation == 'Tanh':
            self.act = torch.nn.Tanh()
        elif activation == 'Hardtanh':
            self.act = torch.nn.Hardtanh(min_val=0, max_val=1)
        elif activation == 'Relu':
            self.act = torch.nn.ReLU()
       
    def forward(self, x, samples=1):
        """
        Forward propagation through the network.
        """
        for layer in self.layers[:-1]:
            x = layer(x, samples)
            x = self.act(x)
        x = self.layers[-1](x,samples)
        x = F.log_softmax(x, dim=2)
        return x
    
    def loss(self, x, target, samples=1):
        """
        Compute the Negative Log Likelihood estimated by Monte Carlo sampling and the kl divergence in close form.
        """
        outputs = self.forward(x, samples=samples)
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction=self.reduction)
        kl = self.kl() / self.num_batches
        return negative_log_likelihood*self.coeff_likeli + kl*self.coeff_kl
    
    def kl(self):
        """
        Compute the KL divergence between the variational distribution and the prior.
        - The coefficient should be 1 in the strict bayesian framework
        - However for cifar we must have coeff<<1 so that we can learn. 
        - Otherwise the regularization is too strong.This is not uncommon in variational methods"
        """
        kl = 0
        for layer in self.layers:
            kl += layer.kl() 
        return kl
    
    def Mean_parameters(self, recurse: bool = True):
        """
        Yield mean parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('mu'):
                yield param

    def Std_parameters(self, recurse: bool = True):
        """
        Yield standard deviation parameters, useful for custom experiments.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if name.endswith('sigma'):
                yield param

