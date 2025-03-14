#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:04:18 2024

@author: djohan
"""
import torch
import numpy as np
import copy
if torch.cuda.is_available():
    device = 'cuda'
    # Print a message indicating that CUDA is available
    print('CUDA is available')
    


class Trainer(object):
    """Object used to train and evaluate the model in different learning scenarios, with different learning algorithm"""
    def __init__(self, learning_scenario, algo_name, optimizer, model, clamp_grad=0):
        super().__init__()
        self.learning_scenario = learning_scenario
        self.algo_name = algo_name
        self.optimizer = optimizer
        self.clamp_grad = clamp_grad
        #coefficient between the likelihood and the prior for BBB. For deep model you usually need to make the KL divergence weaker to be able to learn
        valid_learning_scenarios = {'Classic', 'Task incremental', 'Domain incremental', 'Classic_Animals'}
        valid_algo_names = {'MESU', 'BBB', 'BGD', 'DET', 'BBB_GP', 'EWC', 'SI'}
        if self.learning_scenario not in valid_learning_scenarios:
            raise ValueError(
                "Invalid training name {!r}, should be one of {}".format(
                    self.learning_scenario, valid_learning_scenarios))
        if self.algo_name not in valid_algo_names:
            raise ValueError(
                "Invalid algo name {!r}, should be one of {}".format(
                    self.algo_name, valid_algo_names))
        self.model=model 
        self._print_scenario_details()
        self._print_algo_details()
        if algo_name =='EWC':
            ### It's train the same way, the diff are in the computation of the loss but it is done elsewhere
            self.algo_name = 'DET'
    def _print_scenario_details(self):
        scenario_details = {
            'Classic': 'You chose the "Classic" learning scenario:\n - The model will be trained on one task only.',
            'Task incremental': 'You chose the "Task incremental" learning scenario.\n - The model will be trained on different tasks.\n - One head will correspond to one task.',
            'Domain incremental': 'You chose the "Domain incremental" learning scenario.\n - The model will be trained on different tasks.\n - A single output neuron represents a class for all tasks.\n - In this scenario, we choose to have no task boundaries.'
        }
        print(scenario_details[self.learning_scenario])

    def _print_algo_details(self):
        algo_details = {
            'DET': 'You have selected DET! You will train a determinist neural networks.\n - Make sure your optimizer is from torch.optim library.',
            'MESU': 'You have selected MESU! Refer to: This paper\n - Make sure your optimizer is optimizers.MESU().',
            'BBB': 'You have selected BBB! Refer to: C.Blundell "Weight Uncertainty in Neural Networks" (2015).\n - Make sure your optimizer is from torch.optim library.',
            'BBB_GP': 'You have selected BBB_GP! Refer to: C.Blundell "Weight Uncertainty in Neural Networks" (2015).\n - Make sure your optimizer is from torch.optim library.',
            'BGD': 'You have selected BGD! Refer to: C.Zeno "Task Agnostic Continual Learning Using Online Variational Bayes" (2018).\n - Make sure your optimizer is optimizers.BGD().',
            'EWC': 'You have selected EWC! Refer to Overcoming Catastrophic forgetting.',
            'SI' : 'You have selected SI! F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).'
        }
        print(algo_details[self.algo_name])
       
            
    
    def train_bayes(self,X, Y, batch_size=256, samples_train=10):
        """
        Description:
            Trains a Bayesian neural network classically. 
            MESU and BGD takes into account the prior (KL) into the update rule.
            Therefore the loss is the Negative-Log-Likelihood (NLL). 
            BBB does not takes into account the prior into the update rule.
            Therefore the loss is the free energy (NLL+KL). 
            For consistency with the other models we use a Gaussian prior. 
            Therefore we do not need Monte Carlo sampling to estimates the KL divergence.
    
        Args:
            X: Data for the task.
            Y: Labels for the task.
            batch_size: number of images per update iterations.
            samples_train: Number of Monte Carlo samples (on parameters) during training.
    
        Returns:
            loss_avg: The averaged loss after training on one epoch.
        """
        num_batches=len(X)//batch_size
        self.model.train()
        loss_avg=0
        if self.algo_name.startswith('BBB'):
            self.model.num_batches = num_batches
            
        for batch_idx in range(num_batches):
            
            Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            self.model.zero_grad()
            loss = self.model.loss(Xb, Yb, samples=samples_train)
            loss.backward()
            if self.clamp_grad>0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(),self.clamp_grad)
            self.optimizer.loss = loss.item()
            self.optimizer.step()
            loss_avg += loss.item()/num_batches        
        return loss_avg
    
      
    def train_det(self,X, Y, batch_size=256):
        """
        Description:
            Trains a determinist neural network classically. 
           
        Args:
            X: Data for the task.
            Y: Labels for the task.
            batch_size: number of images per update iterations.
    
        Returns:
            loss_avg: The averaged loss after training on one epoch.
        """
        num_batches=len(X)//batch_size
        self.model.train()
        loss_avg=0
        for batch_idx in range(num_batches):    
            Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            self.model.zero_grad()
            loss = self.model.loss(Xb, Yb)
            loss.backward()
            if self.clamp_grad>0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(),self.clamp_grad)
            self.optimizer.step()
        
            loss_avg += loss.item()/num_batches        
        return loss_avg
    
    def train(self,X, Y, batch_size=256, samples_train=10):
        if self.algo_name!='DET':
            loss = self.train_bayes(X, Y, batch_size=batch_size, samples_train=samples_train)
        else:
            loss = self.train_det(X, Y, batch_size=batch_size)    
        return loss
    
    def train_bayes_domain_inc_continual(self, X0, X1, Y0, Y1, it, batch_size=128, train_epochs=20, ratio=0.75, samples_train=10):
        """
        Description:
            Trains a Bayesian neural network in a domain incremental learning scenario (without task boundaries).
            The proportion of images belong to task 1 and task 2 evolves gradually. 
            At epoch=0 all the images belongs to the first task.
            At epoch=train_epochs all the images belongs to the second task.
            MESU and BGD takes into account the prior (KL) into the update rule.
            Therefore the loss is the Negative-Log-Likelihood (NLL). 
            BBB does not takes into account the prior into the update rule.
            Therefore the loss is the free energy (NLL+KL). 
            For consistency with the other models we use a Gaussian prior. 
            Therefore we do not need Monte Carlo sampling to estimates the KL divergence.
    
        Args:
            X0: Data for the first task.
            X1: Data for the second task.
            Y0: Labels for the first task.
            Y1: Labels for the second task.
            it: Current iteration count.
            batch_size: number of images per update iterations.
            train_epochs: Number of training epochs.
            ratio: For gradually adding images from the new task starting from IT_S iterations to IT_F iterations.
            samples_train: Number of Monte Carlo samples during training.
    
        Returns:
            loss_avg: The averaged loss after training on one epoch.
            it: iteration count that controls the proportion of images belonging to task 1 and task 2. 
        """
        num_batches = len(X0) // batch_size
        self.model.train()
        IT_F = round(len(X0) / batch_size) * train_epochs
        IT_S = round(ratio * len(X0) / batch_size) * train_epochs
        loss_avg=0
        if self.algo_name.startswith('BBB'):
            self.model.num_batches = num_batches
            
        for batch_idx in range(int(num_batches)):
            self.model.zero_grad()
            p = max(0, int(batch_size * (it - IT_S) / (IT_F - IT_S)))
            it += 1
            Xa = X0[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Ya = Y0[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Xb = X1[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Yb = Y1[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            X = torch.cat((Xa[p:], Xb[:p]), axis=0)
            Y = torch.cat((Ya[p:], Yb[:p]), axis=0)
            loss = self.model.loss(X, Y, samples=samples_train)
            loss.backward()
            if self.clamp_grad>0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(),self.clamp_grad)
            self.optimizer.loss=loss.item()
            self.optimizer.step()
            loss_avg += loss/num_batches
            
        return loss_avg, it 
    
    def train_det_domain_inc_continual(self, X0, X1, Y0, Y1, it, batch_size=128, train_epochs=20, ratio=0.75):
        """
        Description:
            Trains a determist neural network in a domain incremental learning scenario (without task boundaries).
            
        Args:
            X0: Data for the first task.
            X1: Data for the second task.
            Y0: Labels for the first task.
            Y1: Labels for the second task.
            it: Current iteration count.
            batch_size: number of images per update iterations.
            train_epochs: Number of training epochs.
            ratio: For gradually adding images from the new task starting from IT_S iterations to IT_F iterations.
    
        Returns:
            loss_avg: The averaged loss after training on one epoch.
            it: iteration count that controls the proportion of images belonging to task 1 and task 2. 
        """
        num_batches = len(X0) // batch_size
        self.model.train()
        IT_F = round(len(X0) / batch_size) * train_epochs
        IT_S = round(ratio * len(X0) / batch_size) * train_epochs
        loss_avg=0
        
            
        for batch_idx in range(int(num_batches)):
            self.model.zero_grad()
            p = max(0, int(batch_size * (it - IT_S) / (IT_F - IT_S)))
            it += 1
            Xa = X0[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Ya = Y0[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Xb = X1[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Yb = Y1[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            X = torch.cat((Xa[p:], Xb[:p]), axis=0)
            Y = torch.cat((Ya[p:], Yb[:p]), axis=0)
            loss = self.model.loss(X, Y)
            loss.backward()
            if self.clamp_grad>0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(),self.clamp_grad)
            self.optimizer.step()
            loss_avg += loss/num_batches
            
        return loss_avg, it 
    
    def train_domain_inc(self, X0, X1, Y0, Y1, it, batch_size=128, train_epochs=20, ratio=0.75, samples_train=10):
        if self.algo_name!='DET':
            loss, it = self.train_bayes_domain_inc( X0, X1, Y0, Y1, it, batch_size=batch_size, train_epochs=train_epochs, ratio=ratio, samples_train=samples_train)
        else:
            loss, it = self.train_det_domain_inc( X0, X1, Y0, Y1, it, batch_size=batch_size, train_epochs=train_epochs, ratio=ratio)
            
        return loss, it 
    
    def train_bayes_task_inc(self, X, Y, batch_size=256, samples_train=10, head=1):
        """
        Description:
            Trains a Bayesian neural network in a task incremental learning scenario (with task boundaries).
            The dataset must corresponds to the head.
            MESU and BGD takes into account the prior (KL) into the update rule.
            Therefore the loss is the Negative-Log-Likelihood (NLL). 
            BBB does not takes into account the prior into the update rule.
            Therefore the loss is the free energy (NLL+KL). 
            For consistency with the other models we use a Gaussian prior. 
            Therefore we do not need Monte Carlo sampling to estimates the KL divergence.
     
        Args:
            X: Data of the task that correspond to the head.
            Y: Labels of the task that correspond to the head.
            batch_size: number of images per update iterations.
            train_epochs: Number of training epochs.
            samples_train: Number of Monte Carlo samples during training.
            head: Indice of the head used in the neural network. 
    
        Returns:
            loss_avg: The averaged loss after training on one epoch.
        """
        num_batches=len(X)//batch_size
        loss_avg=0
        self.model.train()
        if self.algo_name.startswith('BBB'):
            self.model.num_batches = num_batches
        for batch_idx in range(int(num_batches)):
            self.model.zero_grad()
            Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            loss = self.model.loss(Xb, Yb, samples=samples_train, head=head)
            loss.backward()
            if self.clamp_grad>0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(),self.clamp_grad)
            self.optimizer.loss = loss.item()
            self.optimizer.step()
            loss_avg +=loss.item() / num_batches
        return loss_avg
    
    
    def train_det_task_inc(self, X, Y, batch_size=256, head=1):
        """
        Description:
            Trains a determinist neural network in a task incremental learning scenario (with task boundaries).
            The dataset must corresponds to the head.
     
        Args:
            X: Data of the task that correspond to the head.
            Y: Labels of the task that correspond to the head.
            batch_size: number of images per update iterations.
            train_epochs: Number of training epochs.
            head: Indice of the head used in the neural network. 
    
        Returns:
            loss_avg: The averaged loss after training on one epoch.
        """
        num_batches=len(X)//batch_size
        loss_avg=0
        self.model.train()
     
        for batch_idx in range(int(num_batches)):
            self.model.zero_grad()
            Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            loss = self.model.loss(Xb, Yb, head=head)
            loss.backward()
            if self.clamp_grad>0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(),self.clamp_grad)
            self.optimizer.step()
            loss_avg +=loss.item() / num_batches
        return loss_avg
    
    def train_det_synaptic_intelligence_task_inc(self, X, Y, batch_size=12, head=0):
        """
        Train the network using deterministic approximation for the Hessian with synaptic intelligence. Refer to:F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
        Most function are taken from https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/agents/regularization.py
        """
    
        num_batches = len(X) // batch_size
        loss_avg=0
        self.model.train()
        
        for batch_idx in range(num_batches):
            
            old_params = {}
            for i, (name, param) in enumerate(self.model.named_parameters(recurse=True)):
                old_params[name] = param.clone().detach()

            self.model.zero_grad()
            Xb = X[batch_idx * batch_size: (batch_idx + 1) * batch_size].to(device)
            Yb = Y[batch_idx * batch_size: (batch_idx + 1) * batch_size].to(device)
            loss = self.model.loss_nll(Xb, Yb, head)
            loss.backward(retain_graph=True)
            
            unreg_gradients = {}
            for i, (name, param) in enumerate(self.model.named_parameters(recurse=True)):
                if param.grad is not None:
                    unreg_gradients[name] = param.grad.clone().detach()

            self.model.zero_grad()
            Xb = X[batch_idx * batch_size: (batch_idx + 1) * batch_size].to(device)
            Yb = Y[batch_idx * batch_size: (batch_idx + 1) * batch_size].to(device)
            loss = self.model.loss(Xb, Yb, head)
            loss.backward()
            self.optimizer.step()
            
            for i, (name, param) in enumerate(self.model.named_parameters(recurse=True)):
                delta = param.detach() - old_params[name]
                if name in unreg_gradients.keys():
                    self.model.w[name] -= unreg_gradients[name] * delta  # w[idx] is >=0
            loss_avg +=loss.item() / num_batches

        return loss_avg
    
    def train_task_inc(self, X, Y, batch_size=256, samples_train=10, head=1):
        if self.algo_name =='SI':
            loss = self.train_det_synaptic_intelligence_task_inc(X, Y, batch_size=batch_size, head=head)
        elif self.algo_name!='DET':
            loss = self.train_bayes_task_inc(X, Y, batch_size=batch_size, samples_train=samples_train, head=head)
        else:
            loss = self.train_det_task_inc(X, Y, batch_size=batch_size, head=head)
        
        return loss 
            
            
    def eval_bayes(self,X, Y, batch_size=2048, samples_inf=1):
        """
        Description:
            Evaluates a Bayesian neural network trained classically.
    
        Args:
            X: Test data.
            Y: Test labels.
            batch_size: number of images we evaluate at the same time. (To reduce if out of memory)
            samples_inf: Number of Monte Carlo samples during inference.
           
        Returns:
            acc: List of accuracy values for each task.
        """
        num_batches=len(X)//batch_size
        acc=0
        for batch_idx in range(num_batches):
            with torch.no_grad():
                self.model.eval()
                Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                y_pred = torch.exp(self.model.forward(Xb, samples=samples_inf))
                idx_y_pred = torch.mean(y_pred, axis=0).argmax(dim=1)
                acc += len(idx_y_pred[idx_y_pred == Yb]) / (batch_size*num_batches)
        return acc

    
    
    def eval_det(self,X, Y, batch_size=2048):
        """
        Description:
            Evaluates a determinist neural network trained classically.
    
        Args:
            X: Test data.
            Y: Test labels.
            batch_size: number of images we evaluate at the same time. (To reduce if out of memory)
            samples_inf: Number of Monte Carlo samples during inference.
           
        Returns:
            acc: List of accuracy values for each task.
        """
        num_batches=len(X)//batch_size
        acc=0
        for batch_idx in range(num_batches):
            with torch.no_grad():
                self.model.eval()
                Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                y_pred = torch.exp(self.model.forward(Xb))
                idx_y_pred = y_pred.argmax(dim=1)
                acc += len(idx_y_pred[idx_y_pred == Yb]) / (batch_size*num_batches)
        return acc
    
    def Eval(self, X, Y, batch_size=256, samples_inf=1):
        if self.algo_name!='DET':
            acc = self.eval_bayes(X, Y, batch_size=batch_size, samples_inf=samples_inf)
        else:
            acc =self.eval_det(X, Y, batch_size=batch_size)
            
        return acc 
    
            
    def eval_bayes_domain_inc(self, X, Y, PERM, num_task, samples_inf=1):
        """
        Description:
            Evaluates a Bayesian neural network trained in a domain incremental learning scenario.
            for MNIST we can evaluate the entire test dataset in one shot
        Args:
            X: Test data.
            Y: Test labels.
            PERM: List of permutation of MNIST (each permutation correspond to one task).
            num_task: Number of tasks we want to evaluate.
            batch_size: number of images we evaluate at the same time. (To reduce if out of memory)
            head: Indice of the head used in the neural network. 
            
    
        Returns:
            acc: List of accuracy values for the task corresponding to the head.
        """

        with torch.no_grad():
            self.model.eval()
            acc = []
            for j in range(num_task):
                Xb = X[:, PERM[j]].to(device)
                Yb = Y.to(device)
                y_pred = torch.exp(self.model.forward(Xb, samples=samples_inf))
                idx_y_pred = torch.mean(y_pred, axis=0).argmax(dim=1)
                acc_test = len(idx_y_pred[idx_y_pred == Yb]) / len(idx_y_pred)
                acc.append(acc_test)
        return acc
    
    def eval_det_domain_inc(self, X, Y, PERM, num_task):
        """
        Description:
            Evaluates a determinist neural network trained in a domain incremental learning scenario.
            for MNIST we can evaluate the entire test dataset in one shot
        Args:
            X: Test data.
            Y: Test labels.
            PERM: List of permutation of MNIST (each permutation correspond to one task).
            num_task: Number of tasks we want to evaluate.
            batch_size: number of images we evaluate at the same time. (To reduce if out of memory)
            head: Indice of the head used in the neural network. 
            
    
        Returns:
            acc: List of accuracy values for the task corresponding to the head.
        """

        with torch.no_grad():
            self.model.eval()
            acc = []
            for j in range(num_task):
                Xb = X[:, PERM[j]].to(device)
                Yb = Y.to(device)
                y_pred = torch.exp(self.model.forward(Xb))
                idx_y_pred = y_pred.argmax(dim=1)
                acc_test = len(idx_y_pred[idx_y_pred == Yb]) / len(idx_y_pred)
                acc.append(acc_test)
        return acc
    
    def eval_domain_inc(self, X, Y, PERM, num_task, samples_inf=1):
        if self.algo_name!='DET':
            acc = self.eval_bayes_domain_inc(X, Y, PERM, num_task, samples_inf=samples_inf)
        else:
            acc = self.eval_det_domain_inc(X, Y, PERM, num_task)
        
        return acc 
            
            
            
    def eval_bayes_task_inc(self, X, Y, batch_size=2048,samples_inf=1, head=1):
        """
        Description:
            Evaluates a Bayesian neural network trained in a task incremental learning scenario.
            The dataset must corresponds to the head.
            
        Args:
            X: Data of the task that correspond to the head.
            Y: Labels of the task that correspond to the head.
            batch_size: number of images we evaluate at the same time. (To reduce if out of memory)
            samples_inf: Number of Monte Carlo samples during inference.
           
        Returns:
            acc: List of accuracy values for each task.
        """
        num_batches=len(X)//batch_size
        acc=0
        for batch_idx in range(int(num_batches)):
            with torch.no_grad():
                self.model.eval()
                Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                y_pred = torch.exp(self.model.forward(Xb, samples=samples_inf,head=head))
                idx_y_pred = torch.mean(y_pred, axis=0).argmax(dim=1)
                acc += len(idx_y_pred[idx_y_pred == Yb]) / (batch_size*num_batches)
        return acc
    
    def eval_det_task_inc(self, X, Y, batch_size=2048, head=1):
        """
        Description:
            Evaluates a determinist neural network trained in a task incremental learning scenario.
            The dataset must corresponds to the head.
            
        Args:
            X: Data of the task that correspond to the head.
            Y: Labels of the task that correspond to the head.
            batch_size: number of images we evaluate at the same time. (To reduce if out of memory)
            samples_inf: Number of Monte Carlo samples during inference.
           
        Returns:
            acc: List of accuracy values for each task.
        """
        num_batches=len(X)//batch_size
        acc=0
        for batch_idx in range(int(num_batches)):
            with torch.no_grad():
                self.model.eval()
                Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
                y_pred = torch.exp(self.model.forward(Xb, head=head))
                idx_y_pred = y_pred.argmax(dim=1)
                acc += len(idx_y_pred[idx_y_pred == Yb]) / (batch_size*num_batches)
        return acc
    
    def eval_task_inc(self, X, Y, batch_size=2048,samples_inf=1, head=1):
        if self.algo_name  =='SI':
            acc = self.eval_det_task_inc(X, Y, batch_size=batch_size, head=head)
        elif self.algo_name!='DET':
            acc = self.eval_bayes_task_inc(X, Y, batch_size=batch_size,samples_inf=samples_inf, head=head)
        else:
            acc = self.eval_det_task_inc(X, Y, batch_size=batch_size, head=head)
        
        return acc 
        
    def train_det_synaptic_intelligence(self, X, Y, IT, L, batch_size=128):
        """
        Train the network using deterministic approximation for the Hessian with synaptic intelligence. Refer to:F. Zenke "Continual Learning Through Synaptic Intelligence" (2017).
        Most function are taken from https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/agents/regularization.py
        """
        it=0
        it += IT
        if self.algo_name != 'DET':
            raise ValueError("Synaptic intelligence is made for determinist neural networks.")
        num_batches = len(X) // batch_size
        self.model.train()
        
        for batch_idx in range(num_batches):
            unreg_gradients = {}
            
            # 1. Save current parameters
            old_params = {}
            for idx, p in enumerate(self.model.parameters()):
                old_params[idx] = p.clone().detach()
    
            self.model.zero_grad()
            Xb = X[batch_idx * batch_size: (batch_idx + 1) * batch_size].to(device)
            Yb = Y[batch_idx * batch_size: (batch_idx + 1) * batch_size].to(device)
            loss = self.model.loss(Xb, Yb)
            loss.backward(retain_graph=True)
            
            for idx, p in enumerate(self.model.parameters()):
                if p.grad is not None:
                    unreg_gradients[idx] = p.grad.clone().detach()
    
            self.optimizer.step()
            
            for idx, p in enumerate(self.model.parameters()):
                delta = p.detach() - old_params[idx]
                if idx in unreg_gradients.keys():
                    self.model.w[idx] -= unreg_gradients[idx] * delta  # w[idx] is >=0
                    
            it += 1
            if it in L:
                return it
        return it
    
    
    
    def train_bayes_approx_hessian(self,X, Y, IT, L, batch_size=256, samples_train=10):
        """
        Train the network using Bayes for the Hessian approximation.
        """
        it=0
        it += IT
        num_batches = len(X) // batch_size
        self.model.train()
        if self.algo_name == 'DET':
            samples = 0
        else: 
            samples=samples_train
        if self.algo_name.startswith('BBB'):
            self.model.num_batches = num_batches
        for batch_idx in range(int(num_batches)):
            self.model.zero_grad()
            Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            loss = self.model.loss(Xb, Yb, samples=samples)
            loss.backward()
            self.optimizer.loss = loss.item()
            self.optimizer.step()
            it += 1
            if it in L:
                return it
        return it
    
    def train_det_approx_hessian(self,X, Y, IT, L, batch_size=256):
        """
        Train the network using for the Hessian approximation.
        """
        it=0
        it += IT
        num_batches = len(X) // batch_size
        self.model.train()
        
        for batch_idx in range(int(num_batches)):
            self.model.zero_grad()
            Xb = X[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            Yb = Y[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            loss = self.model.loss(Xb, Yb)
            loss.backward()
            self.optimizer.step()
            it += 1
            if it in L:
                return it
        return it

    
        

