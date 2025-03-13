#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:57:18 2024

@author: djohan
"""
import torch
import models
from optimizers import BGD,MESU
import numpy as np
from torch import autograd 



def select_model_and_optim(args_dict):
    """Return the appropriate neural network model and optimizer based on the learning scenario, algorithm, and dataset."""
    
    # Convert args to a dictionary for dynamic unpacking

    # Validation checks
    valid_learning_scenarios = {'Classic', 'Task incremental', 'Domain incremental'}
    valid_algo_names = {'MESU', 'BBB', 'BGD', 'DET', 'BBB_GP', 'EWC', 'SI'}
    valid_datasets = {'MNIST', 'KMNIST', 'CIFAR10', 'CIFAR100', 'CIFAR110','CIFAR20','ANIMALS'}

    if args_dict['learning_scenario'] not in valid_learning_scenarios:
        raise ValueError(f"Invalid learning scenario {args_dict['learning_scenario']!r}, expected one of {valid_learning_scenarios}")
    if args_dict['algo'] not in valid_algo_names:
        raise ValueError(f"Invalid algorithm {args_dict['algo']!r}, expected one of {valid_algo_names}")
    if args_dict['dataset'] not in valid_datasets:
        raise ValueError(f"Invalid dataset {args_dict['dataset']!r}, expected one of {valid_datasets}")
    if len(args_dict['archi_fcnn']) < 2:
        raise ValueError("Invalid FCNN architecture length; expected at least length 2")
  

    # Helper function to choose optimizer
    def get_optimizer(model, args_dict):
        if args_dict['algo'] == 'MESU':
            return MESU(model, args_dict)  # Unpack all args dynamically
        elif args_dict['algo'] == 'BGD':
            return BGD(model, args_dict)   # Unpack all args dynamically
        elif args_dict['torch_optim_name'] == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=args_dict['lr'], weight_decay=args_dict['weight_decay'])
        else:
            return torch.optim.SGD(model.parameters(), lr=args_dict['lr'], weight_decay=args_dict['weight_decay'])
    
        # Model selection
    def select_model(args_dict):
        # MNIST or ANIMALS FC datasets
        if args_dict['dataset'] in {'MNIST', 'ANIMALS'}:
            if args_dict['algo'] == 'BBB':
                return models.FCNN_BBB(args_dict)
            elif args_dict['algo'] == 'BBB_GP':
                return models.FCNN_BBB_GP(args_dict)
            elif args_dict['algo'] == 'DET':
                return models.FCNN_DET(args_dict)
            else:
                return models.FCNN(args_dict)
        

        
        # CIFAR10 dataset
        elif args_dict['dataset'].startswith('CIFAR'):
            if args_dict['learning_scenario']!='Task incremental':
                if args_dict['algo'] == 'BBB':
                    if args_dict['cnn_mixte']:
                        return models.CNN_CIFAR_MIXTE_BBB(args_dict)
                    else:
                        return models.CNN_CIFAR_BBB(args_dict)
                elif args_dict['algo'] == 'BBB_GP':
                    if args_dict['cnn_mixte']:
                        return models.CNN_CIFAR_MIXTE_BBB_GP(args_dict)
                    else:
                        return models.CNN_CIFAR_BBB_GP(args_dict)
                elif args_dict['algo'] == 'DET':
                    return models.CNN_CIFAR_DET(args_dict)
            
                elif args_dict['cnn_mixte']:
                    return models.CNN_CIFAR_MIXTE(args_dict)
                else:
                    return models.CNN_CIFAR(args_dict)
                
            
            elif args_dict['learning_scenario'] == 'Task incremental':
                if args_dict['algo'] == 'BBB':
                    if args_dict['cnn_mixte']:
                        return models.CNN_CIFAR_K_HEAD_MIXTE_BBB(args_dict)
                    else:
                        return models.CNN_CIFAR_K_HEAD_BBB(args_dict)
                elif args_dict['algo'] == 'BBB_GP':
                    if args_dict['cnn_mixte']:
                        return models.CNN_CIFAR_K_HEAD_MIXTE_BBB(args_dict)
                    else:
                        return models.CNN_CIFAR_K_HEAD_BBB_GP(args_dict)
                elif args_dict['algo'] == 'DET':
                    return models.CNN_CIFAR_K_HEAD_DET(args_dict)
                elif args_dict['algo'] == 'EWC':
                    return models.CNN_CIFAR_K_HEAD_EWC(args_dict)
                elif args_dict['algo'] == 'SI':
                    return models.CNN_CIFAR_K_HEAD_SI(args_dict)
                elif args_dict['cnn_mixte']:
                    return models.CNN_CIFAR_K_HEAD_MIXTE(args_dict)
                else:
                    return models.CNN_CIFAR_K_HEAD(args_dict)
       
        else:
            # Handle untested configurations
            raise ValueError("Invalid configuration; this scenario was not tested.")
    model = select_model(args_dict)
    optimizer = get_optimizer(model, args_dict)
    
    return model, optimizer


# def c_sigma(N,batch_size,dataset_size,epochs):
#     num_iterations=(dataset_size//batch_size)*epochs
#     return N//num_iterations

def prediction_bayes(Net, X, Y, batch_size=10000, samples_inf=10):
    """Function used to return not only the accuracy, but also the raw prediction, and the index of the prediction
    It is used to evaluate the performances related to a dataset the model has been trained on """
    num_batches = max(1,len(X) // batch_size)
    acc_test = 0
    Y_pred = torch.zeros(samples_inf, len(Y), Net.num_classes, device='cuda')
    Idx_y_pred = torch.zeros((len(Y)) , dtype=torch.long)
    for batch_idx in range(num_batches):
        with torch.no_grad():
            Net.eval()
            Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to('cuda')
            Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to('cuda')
            y_pred = torch.exp(Net.forward(Xb, samples=samples_inf))
            Y_pred[:, batch_idx * batch_size:(batch_idx + 1) * batch_size, :] = y_pred
            idx_y_pred  = torch.mean(y_pred, dim=0).argmax(dim=1)
            Idx_y_pred[batch_idx * batch_size:(batch_idx + 1) * batch_size] = idx_y_pred
            acc_test += (idx_y_pred == Yb).sum().item() / (batch_size * num_batches)
    
    return acc_test, Y_pred, Idx_y_pred


def prediction_bayes_un(Net, X_un, batch_size=100, samples_inf=10):
    """Function used to return  the raw prediction.
    It is used to evaluate the performances related to a dataset the model has not been trained on.
    Particularly we want to check if the epistemic uncertainty is indeed elavated. There is no point to compute the accuracy.
    However the predictive and aleatoric uncertainty are of interest if we want to check the epistemic uncertainty is indeed the better marker for ood images."""
    num_batches = max(1,len(X_un) // batch_size)
    Y_pred = torch.zeros(samples_inf, len(X_un), Net.num_classes, device='cuda')
    for batch_idx in range(num_batches):
        with torch.no_grad():
            Net.eval()
            Xb = X_un[batch_idx * batch_size:(batch_idx + 1) * batch_size].to('cuda')
            y_pred = torch.exp(Net.forward(Xb, samples=samples_inf))
            Y_pred[:, batch_idx * batch_size:(batch_idx + 1) * batch_size, :] = y_pred
    return Y_pred


def prediction_un(Net, X_un, batch_size=100, samples_inf=10):
    """Function used to return  the raw prediction.
    It is used to evaluate the performances related to a dataset the model has not been trained on.
    Particularly we want to check if the epistemic uncertainty is indeed elavated. There is no point to compute the accuracy.
    However the predictive and aleatoric uncertainty are of interest if we want to check the epistemic uncertainty is indeed the better marker for ood images."""
    num_batches = max(1,len(X_un) // batch_size)
    Y_pred = torch.zeros(len(X_un), Net.num_classes, device='cuda')
    for batch_idx in range(num_batches):
        with torch.no_grad():
            Net.eval()
            Xb = X_un[batch_idx * batch_size:(batch_idx + 1) * batch_size].to('cuda')
            y_pred = torch.exp(Net.forward(Xb))
            Y_pred[batch_idx * batch_size:(batch_idx + 1) * batch_size, :] = y_pred
    return Y_pred


def acc_and_uncertainties(Net, X, Y, batch_size=10000, samples_inf=10):
    """Function used to return the accuracy and the uncertainties.
    It is used to evaluate the performances related to a dataset the model has been trained on.
    """
    acc_test, Yp, idx_y_pred = prediction_bayes(Net, X, Y, batch_size=batch_size, samples_inf=samples_inf)
    predictive = -torch.sum(torch.log(torch.mean(Yp, dim=0)+1e-12) * torch.mean(Yp, dim=0), dim=-1)
    aleatoric = -torch.sum(torch.mean(torch.log(Yp+1e-12) * Yp, dim=0), dim=-1)
    epistemic = predictive - aleatoric
    return predictive, aleatoric, epistemic, acc_test, idx_y_pred 


def uncertainties(Net, X_un, batch_size=10000, samples_inf=10):
    """Function used to return  the uncertainties only.
    It is used to evaluate the performances related to a dataset the model has not been trained on.
    Particularly we want to check if the epistemic uncertainty is indeed elavated. There is no point to compute the accuracy.
    However the predictive and aleatoric uncertainty are of interest if we want to check the epistemic uncertainty is indeed the better marker for ood images."""
    Yp = prediction_bayes_un(Net, X_un, batch_size=batch_size, samples_inf=samples_inf)
    predictive = -torch.sum(torch.log(torch.mean(Yp, dim=0)+1e-12) * torch.mean(Yp, dim=0), dim=-1)
    aleatoric = -torch.sum(torch.mean(torch.log(Yp+1e-12) * Yp, dim=0), dim=-1)
    epistemic = predictive - aleatoric
    return predictive, aleatoric, epistemic

def uncertainties_det(Net, X_un, batch_size=10000):
    """Function used to return  the uncertainties only.
    It is used to evaluate the performances related to a dataset the model has not been trained on.
    Particularly we want to check if the epistemic uncertainty is indeed elavated. There is no point to compute the accuracy.
    However the predictive and aleatoric uncertainty are of interest if we want to check the epistemic uncertainty is indeed the better marker for ood images."""
    Yp = prediction_un(Net, X_un, batch_size=batch_size)
    aleatoric = -torch.sum(torch.log(Yp+1e-12) * Yp, dim=-1)
    return aleatoric

def AUC(FPR, TPR):
    """Compute the Area Under the Curve (AUC) using the trapezoidal rule."""
    return np.trapz(TPR, FPR)

def eval_aleatoric(aleatoric, idx_y_pred, Y):
    """Function used to return the AUC of the aleatoric uncertainty"""
    TPRa, FPRa = [], []
    TPtot = (idx_y_pred == Y).sum().item()
    FPtot = (idx_y_pred != Y).sum().item()
    aleatoric_fp = aleatoric[idx_y_pred != Y]
    sorted_fp = aleatoric_fp.sort().values

    for t in sorted_fp:
        I = idx_y_pred[aleatoric < t]
        TP = (I == Y[aleatoric < t]).sum().item()
        FP = (I != Y[aleatoric < t]).sum().item()

        TPRa.append(TP / TPtot)
        FPRa.append(FP / FPtot)
    AUCA = AUC(FPRa, TPRa)
    return AUCA


def eval_epistemic(epistemic, epistemic_un, X, X_un, step_max=1000):
    """Function used to return the AUC of the epistemic uncertainty"""
    TPRe, FPRe = [], []
    TPtot = len(X)
    FPtot = len(X_un)
    sorted_fp = epistemic_un.sort().values
    step_reduction = max(1,len(sorted_fp)//step_max)
    sorted_fp = sorted_fp[::step_reduction]
    for t in sorted_fp:
        TP = len(X[epistemic < t])
        FP = len(X_un[epistemic_un < t])
        TPRe.append(TP / TPtot)
        FPRe.append(FP / FPtot)      
    AUCE = AUC(FPRe, TPRe)
    return AUCE



def fisher(net, X0, Y0, batch_size):
    """
    Compute and return the Fisher information matrix for the net's layers[-1] layer parameters.
    """
    num_batches = len(X0) // batch_size
    fisher_matrix = torch.zeros(list(net.layers[-1].parameters())[0].shape)

    for batch_idx in range(num_batches):
        loss = net.loss(X0[batch_idx * batch_size: (batch_idx + 1) * batch_size], 
                        Y0[batch_idx * batch_size: (batch_idx + 1) * batch_size]) 
        
        # Clear previous gradients
        net.zero_grad()
        
        # Compute gradients of the loss w.r.t the parameters
        grads = autograd.grad(loss, list(net.layers[-1].parameters())[0], create_graph=True)[0]
        
        # Accumulate the outer product of gradients into the Fisher matrix
        # We divide by num_batches as the reduction='mean' for deterministic neural network
        fisher_matrix += grads.detach().pow(2) / num_batches
        
    return fisher_matrix.cpu().detach().numpy().flatten()


def hessian_det(net, X0, Y0, batch_size):
    """
    Compute and return the Hessian for the net's layers[-1] layer parameters concerning the mean.
    """
    num_batches = len(X0) // batch_size
    H = torch.zeros(list(net.layers[-1].parameters())[0].shape)

    for batch_idx in range(num_batches):
        net.zero_grad()
        loss = net.loss(X0[batch_idx * batch_size: (batch_idx + 1) * batch_size], 
                        Y0[batch_idx * batch_size: (batch_idx + 1) * batch_size])
        
        grads = autograd.grad(loss, list(net.layers[-1].parameters())[0], create_graph=True)[0]
        for i in range(len(H)):
            for j in range(len(H[0])):
                g2 = autograd.grad(grads[i, j], list(net.layers[-1].parameters())[0], create_graph=True)
                # We divide by num_batches as the reduction='mean' for deterministic neural network
                H[i, j] += g2[0][i, j].detach() / num_batches
                
    return H.cpu().detach().numpy().flatten() 

def hessian_mu(net, X0, Y0, batch_size, samples_train):
    """
    Compute and return the Hessian for the net's layers[-1] layer parameters concerning the mean.
    """
    num_batches = len(X0) // batch_size
    H_mu = torch.zeros(list(net.layers[-1].parameters())[1].shape)

    for batch_idx in range(num_batches):
        net.zero_grad()
        loss = net.loss(X0[batch_idx * batch_size: (batch_idx + 1) * batch_size], 
                        Y0[batch_idx * batch_size: (batch_idx + 1) * batch_size], samples=samples_train)
        grads_mu = autograd.grad(loss, list(net.layers[-1].parameters())[1], create_graph=True)[0]
        for i in range(len(H_mu)):
            for j in range(len(H_mu[0])):
                g2_mu = autograd.grad(grads_mu[i, j], list(net.layers[-1].parameters())[1], create_graph=True)
                H_mu[i, j] += g2_mu[0][i, j].detach() / num_batches
    return  H_mu.cpu().detach().numpy().flatten()


def hessian(net, X0, Y0, batch_size, samples_hessian, PATH):
    """
    Compute and save the full Hessian for the net's layers[-1] layer parameters.
    """
    num_batches = len(X0) // batch_size
    H_mu = torch.zeros(list(net.layers[-1].parameters())[1].shape)
    H_sig = torch.zeros(list(net.layers[-1].parameters())[0].shape)

    for batch_idx in range(num_batches):
        net.zero_grad()
        loss = net.loss(X0[batch_idx * batch_size: (batch_idx + 1) * batch_size], 
                        Y0[batch_idx * batch_size: (batch_idx + 1) * batch_size], samples=samples_hessian)
        
        grads_mu = autograd.grad(loss, list(net.layers[-1].parameters())[1], create_graph=True)[0]
        grads_sig = autograd.grad(loss, list(net.layers[-1].parameters())[0], create_graph=True)[0]

        for i in range(len(H_mu)):
            for j in range(len(H_mu[0])):
                g2_mu = autograd.grad(grads_mu[i, j], list(net.layers[-1].parameters())[1], create_graph=True)
                H_mu[i, j] += g2_mu[0][i, j].detach() / num_batches
                g2_sig = autograd.grad(grads_sig[i, j], list(net.layers[-1].parameters())[0], create_graph=True)
                H_sig[i, j] += g2_sig[0][i, j].detach() / num_batches

    return H_mu.cpu().detach().numpy().flatten(), H_sig.cpu().detach().numpy().flatten()

