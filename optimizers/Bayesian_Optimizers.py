#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:10:54 2024

@author: Dr Djo ;) 
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple
from torch.nn import Module
import torch.nn.init as init
import copy

__all__ = ['MESU', 'mesu', 'BGD', 'bgd']


class MESU(object):
    r"""
    Implements Metaplasticity from Synaptic Uncertainty without alterations.

    Args:
        model: The entire Model parameters. We iterate over the models params that may or may not have gradients. 
        sigma_prior: Standard deviation of the prior over the weights, typically 0.01**0.5, or 0.001**0.5 . Can vary per group.
        N: Number of batches to retain in synaptic memory in the Bayesian Forgetting framework. N acts as a weight between prior and likelihood. It should be DATASET_SIZE//BATCH_SIZE, but using larger values can yield better results.
        clamp_sigma: If >0, clamps the  sigma to this value times sigma_prior, typically 1e-3 or 1e-2.

    Raises:
        ValueError: If input arguments are invalid.

    Note:
        Additional parameters can accelerate learning, such as:
            - Extra learning rates for mu and sigma
            - Individual priors for each synapse
    """

    def __init__(self, model, args_dict):

        
        super().__init__()
        self.model=model
        self.mu_prior=args_dict['mu_prior']
        self.sigma_prior=args_dict['sigma_prior']
        self.N=args_dict['N']
        self.c_sigma=args_dict['c_sigma']
        self.c_mu=args_dict['c_mu']
        self.second_order=args_dict['second_order']
        self.clamp_sigma = args_dict['clamp_sigma']
        self.clamp_mu = args_dict['clamp_mu']
        self.ratio_max = args_dict['ratio_max']
        self.moment_sigma = args_dict['moment_sigma']
        self.moment_mu = args_dict['moment_mu']
        self.loss = 0
        self.num_params = len(list(model.parameters())) 
        print(f'Optimizer initialized with {self.num_params} Gaussian variational Tensor parameters.')
        self.grad_eff_sigma={}
        self.hess_approx={}
        self.grad_eff_mu={}
        with torch.no_grad():
            for i, (name, param) in enumerate(model.named_parameters(recurse=True)):
                if name.endswith('sigma'):
                   self.grad_eff_sigma[name]=torch.zeros_like(param.data)
                if name.endswith('mu'):
                   self.grad_eff_mu[name]=torch.zeros_like(param.data)
                
    def step(self,):
        """Performs a single optimization step."""
                   
        mesu(model=self.model,
             grad_eff_sigma = self.grad_eff_sigma,
             grad_eff_mu = self.grad_eff_mu,
             hess_approx = self.hess_approx,
             moment_sigma = self.moment_sigma,
             moment_mu = self.moment_mu,
             mu_prior = self.mu_prior,
             sigma_prior=self.sigma_prior,
             N=self.N,
             c_sigma=self.c_sigma,
             c_mu=self.c_mu,
             second_order = self.second_order,
             clamp_sigma = self.clamp_sigma,
             clamp_mu = self.clamp_mu,
             ratio_max = self.ratio_max,
             loss = self.loss
            )



def mesu(model: Module, *, grad_eff_sigma:Tensor,grad_eff_mu:Tensor, hess_approx:Tensor, moment_sigma: float, moment_mu: float, mu_prior:float, sigma_prior: float, N: int, c_sigma:float, c_mu:float, second_order:bool, clamp_sigma:list, clamp_mu:list, ratio_max:float, loss:float):
    num_params_tot = sum(p.numel() for p in model.parameters())
    previous_param = None
    for i, (name, param) in enumerate(model.named_parameters(recurse=True)):
        num_params_layer = param.numel()
        if name.endswith('sigma'):
            sigma=param
            variance=param.data**2
            grad_sigma=param.grad
            name_sigma = name
        if name.endswith('mu'):
            mu=param
            grad_mu=param.grad
            name_mu = name
            if grad_sigma!=None and grad_mu!=None:
                ### MOMENTUM ANS DENOMINATOR ARE NOT USED IN MESU
                grad_eff_sigma[name_sigma].mul_(moment_sigma)
                grad_eff_sigma[name_sigma].add_((1-moment_sigma)*grad_sigma)
                grad_eff_mu[name_mu].mul_(moment_mu)
                grad_eff_mu[name_mu].add_((1-moment_mu)*grad_mu)
                denominator = 1 + second_order * sigma * grad_eff_sigma[name_sigma].abs()
                delta_sigma = -c_sigma*(variance * grad_eff_sigma[name_sigma] + sigma.data * (variance-sigma_prior ** 2) / (N* (sigma_prior** 2))) / denominator
                delta_mu = -c_mu*(variance* grad_eff_mu[name_mu] + variance * (mu.data-mu_prior) / (N * sigma_prior **2)) / denominator
                delta_sigma = torch.clamp(delta_sigma,-ratio_max*sigma.data,ratio_max*sigma.data)
                delta_mu = torch.clamp(delta_mu,-ratio_max*sigma.data,ratio_max*sigma.data)
                sigma.data.add_(delta_sigma)
                mu.data.add_(delta_mu)
                
        
            if grad_sigma==None and grad_mu!=None:
                #When we have determinist convolution we want to use mesu det we will need hess approx :) be carefull in that case we need the loss in the optimizer...
                loss_approx = 0.5* ((grad_mu.data**2)*variance).mean()
                hess_approx_ = loss_approx/((variance) * loss+1e-2)
                # Now we can update sigma based on the relation we have betwwen the gradient over sigma and H 
                grad_sigma = hess_approx_*sigma.data  
                denominator = 1 + second_order * sigma * grad_sigma
                delta_sigma = -c_sigma*(variance * grad_sigma + sigma.data * (variance-sigma_prior ** 2) / (N* (sigma_prior** 2))) / denominator
                delta_mu = -c_mu*(variance* grad_mu + variance * (mu.data-mu_prior) / (N * sigma_prior **2)) / denominator
                delta_sigma = torch.clamp(delta_sigma,-ratio_max*sigma.data,ratio_max*sigma.data)
                delta_mu = torch.clamp(delta_mu,-ratio_max*sigma.data,ratio_max*sigma.data)
                sigma.data.add_(delta_sigma)
                mu.data.add_(delta_mu)
                

class BGD(object):
    r"""
    Implements Bayesian Gradient Descent based on the work of Chen Zeno --> https://arxiv.org/pdf/1803.10123.

    Args:
        params: Model parameters. Parameters representing 'sigma' must be defined before 'mu'.
        learning_rate: It should be 1. Can be greater to adjust the convergence rate. 

    Raises:
        ValueError: If input arguments are invalid.
    """
    def __init__(self, model, args_dict):
    
        
        super().__init__()
        self.model=model
        self.c_sigma=args_dict['c_sigma']
        self.c_mu=args_dict['c_mu']
        self.clamp_sigma = args_dict['clamp_sigma']
        self.clamp_mu = args_dict['clamp_mu']
        self.ratio_max= args_dict['ratio_max']
        self.loss = 0
        num_params = len(list(model.parameters())) 
        print(f'Optimizer initialized with {num_params} Gaussian variational Tensor parameters.')
        self.hess_approx={}
        for i, (name, param) in enumerate(model.named_parameters(recurse=True)):
            if name.endswith('sigma'):
                self.hess_approx[name]=torch.ones_like(param)/param**2 -1
        
                

    def step(self):
        """Performs a single optimization step."""
        
        bgd(model=self.model,
            hess_approx = self.hess_approx, 
            c_sigma=self.c_sigma,
            c_mu=self.c_mu,
            clamp_sigma = self.clamp_sigma,
            clamp_mu = self.clamp_mu,
            ratio_max = self.ratio_max,
            loss = self.loss
                )

def bgd(model: Module, *, hess_approx:Tensor, c_sigma:float, c_mu:float, clamp_sigma:list, clamp_mu:list, ratio_max:float, loss:float):
    previous_param = None
    num_params_tot = sum(p.numel() for p in model.parameters())
    for i, (name, param) in enumerate(model.named_parameters(recurse=True)):
        num_params_layer = param.numel()
        if name.endswith('sigma'):
            sigma=param
            variance = sigma.data**2         
            grad_sigma=param.grad
            name_sigma = name
        if name.endswith('mu'):
            mu=param
            grad_mu=param.grad
            if grad_sigma!=None and grad_mu!=None:
                delta_sigma = c_sigma*(-0.5 * (sigma.data**2) * grad_sigma + sigma.data * (-1 + (1 + 0.25 * ((sigma.data**2) * (grad_sigma ** 2))) ** 0.5))
                delta_mu = c_mu*(-(sigma.data**2) * grad_mu)
                delta_sigma = torch.clamp(delta_sigma,-ratio_max*sigma.data,ratio_max*sigma.data)
                delta_mu = torch.clamp(delta_mu,-ratio_max*sigma.data,ratio_max*sigma.data)
                sigma.data.add_(delta_sigma)
                mu.data.add_(delta_mu)
               
            if grad_sigma==None and grad_mu!=None:
                #When we have determinist convolution we want to use mesu det we will need hess approx :) be carefull in that case we need the loss in the optimizer...
                loss_approx = 0.5* ((grad_mu.data**2)*variance).mean()
                hess_approx_ = loss_approx/(variance * loss+1e-2)
                # Now we can update sigma based on the relation we have betwwen the gradient over sigma and H 
                grad_sigma = hess_approx_*sigma.data
                # Now we can update sigma based on the relation we have betwwen the gradient over sigma and H 
                delta_sigma = c_sigma*(-0.5 * variance * (grad_sigma) + sigma.data * (-1 + (1 + 0.25 * (variance * ((grad_sigma) ** 2))) ** 0.5))
                delta_mu = c_mu*(-variance* grad_mu)
                delta_sigma = torch.clamp(delta_sigma,-ratio_max*sigma.data,ratio_max*sigma.data)
                delta_mu = torch.clamp(delta_mu,-ratio_max*sigma.data,ratio_max*sigma.data)
                sigma.data.add_(delta_sigma)
                mu.data.add_(delta_mu)
                
