#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:31:13 2024

@author: djohan
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:02:27 2023

@author: DB262466
"""


import math
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
import numpy as np

# NOT USED IN THE MAIN ARTICLE, THIS IS TO IMPLEMENT BAYES BY BACKPROP
# I LEAVE IT HERE IN CASE SOMEONE IS INTERESTED


# Check if CUDA is available
if torch.cuda.is_available():
    # Set the default data type for tensors
    torch.set_default_dtype(torch.float32)
    # Set the default device to CUDA (GPU)
    torch.set_default_device('cuda')
    # Print a message indicating that CUDA is available

 
class Gaussian_BBB_GP(object):
    """Object that : 
        - This object use a variable change, as proposed originally in BBB to prevent the risk of sigma<0
        - use the reparametrisation tricks to sample the weights of a linear layer. 
        - add a dimension in first dimmension for weights and biases samples """
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu  # Mean of the distribution
        self.rho = rho  # Standard deviation of the distribution
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self, samples=1):
        # option to act as a determinist convolution
        if samples == 0:
            # Use the mean value for inference
            return self.mu.unsqueeze(0)
        else:
            # Sample from the standard normal and adjust with sigma and mu
            sigma = self.sigma.unsqueeze(0).repeat(samples, *([1]*len(self.mu.shape)))
            epsilon = torch.empty_like(sigma).normal_()
            mu = self.mu.unsqueeze(0).repeat(samples, *([1]*len(self.mu.shape)))
            return mu + sigma * epsilon
        
    

    
class Linear_BBB_GP(Module):
    """Bayesian linear layer.
    This module is primarly built for BBB with a gaussian prior. 
    BBB use a variable change for sigma.
    When using a gaussian prior the KL divergence between the prior and the variational distribution can be computed in close form.
    Almost identical to pytorch module, except weights and biases are gaussian.
    """
    __constants__ = ['in_features', 'out_features']
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, zeroMean: bool = False, sigma_init=0.1, sigma_prior=0.1, mu_prior=0, device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_BBB_GP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define weight parameters
        self.weight_rho = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_mu = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bound =  math.sqrt(2/in_features)
        self.sigma_init = sigma_init
        self.sigma_prior = sigma_prior
        self.mu_prior = mu_prior
        
        self.weight = Gaussian_BBB_GP(self.weight_mu, self.weight_rho)
        # Control for zero mean initialization
        self.zeroMean = zeroMean
        
        # Define bias if applicable, sigma must be define before mu to be compatible with Bayesian_Optimizers.py
        if bias:
            self.bias_rho = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_mu = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias = Gaussian_BBB_GP(self.bias_mu, self.bias_rho)
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the parameters."""
        if self.zeroMean:
            # put all the mean value to zero
            init.constant_(self.weight_mu, 0)
        else:
            #like kaiming uniform initialisation
            init.uniform_(self.weight_mu, -self.bound, self.bound)
        rho_init = np.log(np.exp(self.sigma_init)-1)
        init.constant_(self.weight_rho, rho_init)
        if self.bias is not None:
            #bias mean value always intialize to zero
            init.constant_(self.bias_mu,0)
            init.constant_(self.bias_rho, rho_init)
            

    def forward(self, x: Tensor, samples: int) -> Tensor:
        """Forward pass using sampled weights and biases.
        We have three dimmensions for the weights samples, the shape is (samples,out_features,in_features) denoted in the einstein sum as soi
        We habe three dimmensions for the inputs  samples, the shape is (samples,batch_size,in_features) denoted in the einstein sum as sbi
        We habe three dimmensions for the outputs  samples, the shape is (samples,batch_size,out_features) denoted in the einstein sum as sbo"""
        samples_dim = np.maximum(samples,1)
        if x.dim() == 2:
            # This is the case for the input image, it will be reshape to (1, batch_size,  W)
            # Each intermediate features map will have 3 dimensions (samples, batch_size, W)
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            # This is the case for the input image, we need to copy it to have the necessary shape : (samples, batch_size, W)
            x = x.repeat(samples_dim,1,1)
        W = self.weight.sample(samples)
        if self.bias:
            B = self.bias.sample(samples)
            return torch.einsum('soi, sbi -> sbo', W, x) + B[:, None]
        else:
            return torch.einsum('soi, sbi -> sbo', W, x)
   
    def kl(self,):
        """Analytical form of the KL. 
            - The constant term -1/2 is not added becasue we care only about the derivative
            - The prior is a multivariate gaussian centered in zero with a std equal to the attribute sigma_prior
        """
            
        kl= (torch.log(self.sigma_prior/self.weight.sigma) + 0.5*(self.weight.sigma**2 + (self.weight_mu-self.mu_prior)**2)/(self.sigma_prior**2)).sum()
        if self.bias is not None:
            kl += (torch.log(self.sigma_prior/self.bias.sigma) + 0.5*(self.bias.sigma**2 + (self.bias_mu-self.mu_prior)**2)/(self.sigma_prior**2)).sum()
        return kl
    
   
    def extra_repr(self) -> str:
        """Representation for pretty print and debugging."""
        return 'in_features={}, out_features={}, sigma_init={}, sigma_prior={}, bias={}'.format(
            self.in_features, self.out_features, self.sigma_init, self.sigma_prior, self.bias is not None)





class ConvNd_BBB_GP(Module):
    """Bayesian convolutional layer, for bayes by backprop with gaussian prior.
    This module is primarly built for BBB with a gaussian prior. BBB use a variable change for sigma.
    When using a gaussian prior the KL divergence between the prior and the variational distribution can be computed in close form.
    Almost identical to pytorch module, except weights and biases are gaussian, and each group in input and output channel represent a Monte Carlo sample.
    """
    
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                      'padding_mode', 'output_padding', 'in_channels',
                      'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    samples: int #replace groups in F.conv2d, weights samples are concatenated across the first dimension
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    sigma_init: float
    sigma_prior: float


    def __init__(self,
                  in_channels: int,
                  out_channels: int,
                  kernel_size: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  padding: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  transposed: bool,
                  output_padding: Tuple[int, ...],
                  sigma_init: float,
                  sigma_prior: float,
                  bias: bool,
                  padding_mode: str,
                  device=None,
                  dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConvNd_BBB_GP, self).__init__()
       

        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        self.sigma_init = sigma_init 
        self.sigma_prior = sigma_prior
        
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                    range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
            
        if transposed:
            self.weight_rho = Parameter(torch.empty(
                (in_channels, out_channels, *kernel_size), **factory_kwargs))
            self.weight_mu = Parameter(torch.empty(
                (in_channels, out_channels, *kernel_size), **factory_kwargs))
            
        else:
            self.weight_rho = Parameter(torch.empty(
                (out_channels, in_channels, *kernel_size), **factory_kwargs))
            self.weight_mu = Parameter(torch.empty(
                (out_channels, in_channels, *kernel_size), **factory_kwargs))
           
        self.weight=Gaussian_BBB_GP(self.weight_mu,self.weight_rho)
        if bias:
            self.bias_rho = Parameter(torch.empty(out_channels, **factory_kwargs))
            self.bias_mu = Parameter(torch.empty(out_channels, **factory_kwargs))
            self.bias=Gaussian_BBB_GP(self.bias_mu,self.bias_rho)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight_mu)
        self.bound = math.sqrt(6/(fan_in+fan_out))
        init.uniform_(self.weight_mu, -self.bound, self.bound)
        rho_init = np.log(np.exp(self.sigma_init)-1)
        init.constant_(self.weight_rho, rho_init)
        if self.bias is not None:
            if fan_in != 0:
                init.constant_(self.bias_mu, 0)
                init.constant_(self.bias_rho, rho_init)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
              ', stride={stride}, sigma_init={sigma_init:.3f}, sigma_prior={sigma_prior:.3f}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(ConvNd_BBB_GP, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'



  

class Conv2d_BBB_GP(ConvNd_BBB_GP):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        sigma_init: float = 0.001 ** 0.5,
        sigma_prior: float = 0.01 ** 0.5,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        cnn_sampling: str = 'weights',
        device=None,
        dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d_BBB_GP, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            transposed=False, output_padding=_pair(0), sigma_init=sigma_init, sigma_prior=sigma_prior, bias=bias, padding_mode=padding_mode, **factory_kwargs)
        self.cnn_sampling=cnn_sampling
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor],samples=1):
        """ To compute the monte carlo sampling in an optimum manner, we use the functionality group of F.conv2d """
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, groups=samples)
        return F.conv2d(x, weight, bias, self.stride,
                        self.padding, self.dilation, groups=samples)
    def kl(self,):
        """This function is used only for BBB_GP. 
            - Because the KL divergence is not taken into account with the update rule
            - The constant term -1/2 is not added becasue we care only about the derivative
            - The prior is a multivariate gaussian centered in zero with a std equal to the attribute sigma_prior
           """
        kl= (torch.log(self.sigma_prior/self.weight.sigma) + 0.5*(self.weight.sigma**2 + self.weight_mu**2)/(self.sigma_prior**2)).sum()
        if self.bias is not None:
            kl += (torch.log(self.sigma_prior/self.bias.sigma) + 0.5*(self.bias.sigma**2 + self.bias_mu**2)/(self.sigma_prior**2)).sum()
        return kl
    
    def forward(self, x: Tensor, samples=1) -> Tensor:
        """ The shapings are necessary to take into account that:
            - feature map are made of monte carlo samples, one per group 
            - the input image feed to the neural network has no sampling dimension"""
        if self.cnn_sampling=='weights':
            W= self.weight.sample(samples)
            W = W.view(W.size(0)*W.size(1), W.size(2), W.size(3), W.size(4))
            samples_dim = np.maximum(samples,1)
            x = x.view(x.size(0)//samples_dim,samples_dim*x.size(1),x.size(2),x.size(3))
    
            if self.bias:
                B = self.bias.sample(samples)
                B = B.flatten()
            else:
                B=None
            out = self._conv_forward(x, W,B,samples=samples_dim)
            out = out.view(samples_dim*out.size(0),out.size(1)//samples_dim,out.size(2),out.size(3))
            return out
        elif self.cnn_sampling=='neurons':
            # Reshape the weights
            sigma = self.weight_sigma.unsqueeze(0).repeat(samples, *([1]*len(self.weight_mu.shape)))
            mu = self.weight_mu.unsqueeze(0).repeat(samples, *([1]*len(self.weight_mu.shape)))
            sigma = sigma.view(sigma.size(0)*sigma.size(1), sigma.size(2), sigma.size(3), sigma.size(4))
            mu = mu.view(mu.size(0)*mu.size(1), mu.size(2), mu.size(3), mu.size(4))
            samples_dim = np.maximum(samples,1)
            x = x.view(x.size(0)//samples_dim,samples_dim*x.size(1),x.size(2),x.size(3))

            if self.bias:
                # Each bias samples will be applied to a group in the output channels
                Bsigma = self.bias_sigma.unsqueeze(0).repeat(samples, *([1]*len(self.weight_mu.shape)))
                Bmu = self.bias_mu.unsqueeze(0).repeat(samples, *([1]*len(self.weight_mu.shape)))
                Bsigma = Bsigma.flatten()
                Bmu = Bmu.flatten()

            else:
                B=None
            out_mu = self._conv_forward(x, mu,Bmu,samples=samples_dim)
            out_sigma = self._conv_forward(x**2, sigma**2,Bsigma**2,samples=samples_dim)
            out = (out_mu + torch.randn(out_mu.shape)*(out_sigma**0.5))
            # For the next layer this intermediate features map need to have the shape (samples, batch_size, C, H, W)
            out = out.view(samples_dim*out.size(0),out.size(1)//samples_dim,out.size(2),out.size(3))
            
            return out
        else:
            raise ValueError("cnn_sampling must be one of (neurons,weights)")