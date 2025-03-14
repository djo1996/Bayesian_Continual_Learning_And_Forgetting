#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:59:47 2024

@author: djohan
"""

import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import copy 
from collections import defaultdict


# Function to save arguments to a specified file
def save_arguments_to_file(directory, args, filename):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

# Argument parser setup
parser = argparse.ArgumentParser(description='Simulation Arguments')

# Main simulation parameters
parser.add_argument('--learning_scenario', type=str, required=True,
                    help='Simulation type: Classic, Domain incremental, Task incremental')
parser.add_argument('--dataset', type=str, required=True,
                    help='Dataset: MNIST, CIFAR10, CIFAR100, ANIMALS CONV, ANIMALS FC')
parser.add_argument('--input_transformation', type=str, default="None",
                    help='Choose between normalise or standardise')
parser.add_argument('--shift', type=float, default=0, help='Shift the input distribution')
parser.add_argument('--scale', type=float, default=1, help='Scale the input distribution')
parser.add_argument('--discretize_levels', type=int, default=0, 
                    help='If discretize_levels>0 it will discretize the input with ''discretize_levels'' level between 0 and 1. Be carefull to transform the dataset arcordingly')
parser.add_argument('--result_dir', type=str, required=True, help='Directory to store results')
parser.add_argument('--argfile', type=str, required=True, help='File to store the arguments')
parser.add_argument('--random_seed', type=int, default=7, help='Random seed for reproducibility')
parser.add_argument('--moy_over', type=int, default=10, help='Number of repetitions for simulation')
parser.add_argument('--hessian_approx', type=str, default="None", 
                    help='Hessian approximation method: Fisher, SI, MetaBayes')
parser.add_argument('--parameter_search', type=str, default="None", 
                    help='parameter search')
parser.add_argument('--boundary', type=str, default="Clear", 
                    help='For differnt scenario in task incremental')
parser.add_argument('--seasons', type=int, default=2, 
                    help='if we have unclear task boundary, seasons tells in how many split the task are separated')
# Training parameters
parser.add_argument('--train_epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')
parser.add_argument('--batch_size_fisher', type=int, default=1, help='Training batch size')
parser.add_argument('--batch_size_inf', type=int, default=10000, help='Inference batch size')
parser.add_argument('--samples_train', type=int, default=3, help='Number of MC samples during training')
parser.add_argument('--samples_inf', type=int, default=20, help='Number of MC iterations during testing')

# Model parameters
parser.add_argument('--reduction', type=str, default="sum", help='Loss reduction: sum or mean')
parser.add_argument('--algo', type=str, default="MESU", help='Algorithm: MESU, BGD, BBB, BBB_GP, Det, EWC')
parser.add_argument('--cnn_mixte', type=int, default=0, help='Full bayesian or bayesian fully connected layer only')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes we have to classify')
parser.add_argument('--num_heads', type=int, default=0, help='Number of head in the DNN')
parser.add_argument('--archi_fcnn', type=int, nargs='+', default=[784, 512, 10], help='Fully connected layer architecture')
parser.add_argument('--activation', type=str, default='Relu', help='Activation function')
parser.add_argument('--cnn_sampling', type=str, default='weights', help='Sampling strategy: weights or neurons')

# Optimization parameters
parser.add_argument('--lambda', type=float, default=5, help='coefficient in EWC loss')
parser.add_argument('--c_si', type=float, default=0.1, help='coefficient in SI loss')
parser.add_argument('--damping_factor', type=float, default=0.001, help='damping_factor in SI importance calculation')

parser.add_argument('--coeff_likeli', type=float, default=1, help='Likelihood coefficient (reduction=mean)')
parser.add_argument('--coeff_kl', type=float, default=1, help='KL divergence coefficient for BBB and BBB_GP')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay (torch.optim optimizer)')
parser.add_argument('--torch_optim_name', type=str, default='SGD', help='Optimizer: SGD or Adam')
parser.add_argument('--clamp_grad', type=float, default=0, help='Clamp gradient for BGD or MESU')
parser.add_argument('--c_mu', type=float, default=1, help='Gradient coefficient for mu')
parser.add_argument('--c_sigma', type=float, default=1, help='Gradient coefficient for sigma')
parser.add_argument('--second_order', type=int, default=0, help='If one or more : apply MESU at second order')
parser.add_argument('--clamp_mu', type=float, nargs='+', default=[0,0], help='Clamp mu between the two values of the list, if the fisrt is different from zero')
parser.add_argument('--clamp_sigma', type=float, nargs='+', default=[0,0], help='Clamp mu between the two values of the list, if the fisrt is different from zero')

# Prior and initialization
parser.add_argument('--sigma_init', type=float, default=0.06, help='Initial sigma for fully connected layers')
parser.add_argument('--sigma_init_conv', type=float, default=0.001**0.5, help='Initial sigma for convolutional layers')
parser.add_argument('--sigma_prior', type=float, default=0.06, help='Prior standard deviation (MESU, BBB_GP)')
parser.add_argument('--mu_prior', type=float, default=0.0, help='Prior mean (MESU, BBB_GP)')
parser.add_argument('--N', type=float, default=1e6, help='Number of batches to retain for sigma')
parser.add_argument('--ratio_max', type=float, default=0.1, help='Maximum value of delta_sigma/sigma or delta_mu/sigma')
parser.add_argument('--moment_sigma', type=float, default=0., help='If you want to add some momentum, not used in the paper')
parser.add_argument('--moment_mu', type=float, default=0., help='If you want to add some momentum, not used in the paper')

# Task parameters
parser.add_argument('--num_task', type=int, default=51, help='Number of permutations for Permuted MNIST')
parser.add_argument('--ratio', type=float, default=0.99, help='Gradual addition of images from new tasks')
parser.add_argument('--train_epochs_A', type=int, default=5, help='Training epochs for first task in Split CIFAR110')
parser.add_argument('--train_epochs_B', type=int, default=1, help='Training epochs for the other tasks in Split CIFAR110')

# Additional parameters for Hessian diagonal approximation
parser.add_argument('--L', type=float, nargs='+', default=[1e3, 1e4, 1e5, 1e6], help='Hessian diagonal approximation levels')


# Parse the arguments
args = parser.parse_args()

# Save the arguments to the specified file in the result directory
save_arguments_to_file(args.result_dir, args, args.argfile)

# Set seeds for reproducibility
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.random_seed)

    
from dataloader import DataLoader
from trainer import Trainer
import utils     
    
# Define the right model and optimizer for the experiment
model,optimizer = utils.select_model_and_optim(vars(args))
DL = DataLoader(vars(args))
X_train=DL.X_train
X_test=DL.X_test
Y_train=DL.Y_train
Y_test=DL.Y_test

# Define the right Trainer
TR = Trainer(args.learning_scenario, args.algo, optimizer, model, clamp_grad=args.clamp_grad)

    
 
# USED FOR FIGURE 5   
if args.learning_scenario == 'Task incremental' and args.dataset=='CIFAR110' and args.boundary=="Clear":
    Acc_train=[]
    Acc_test=[]
    with tqdm(total=args.train_epochs_A + args.train_epochs_B*(args.num_heads-1), desc='Training Progress', unit='epoch') as pbar:
        for task in range(args.num_heads):
            if task ==0:
                train_epochs = args.train_epochs_A
            else:
                train_epochs = args.train_epochs_B
            if args.algo=='SI':
                TR.model.create_si_params()
            for epoch in range(train_epochs):
                order = np.random.permutation(len(X_train[task]))
                X_train[task]=X_train[task][order]
                Y_train[task]=Y_train[task][order]
                start_time = time.time()
                loss=TR.train_task_inc(X_train[task],Y_train[task],head=task, batch_size=args.batch_size, samples_train=args.samples_train)
                if (epoch+1)%10==0:
                    acc_test=[]
                    for i in range(args.num_heads):
                        acc_test.append(TR.eval_task_inc(X_test[i],Y_test[i],head=i, batch_size=args.batch_size_inf, samples_inf=args.samples_inf))
                    
                    Acc_test.append(acc_test)
                    end_time = time.time()
                    duration = end_time - start_time
                    duration = round(duration,2)
                    pbar.set_postfix({
                        'acc_test': f'{acc_test[task]:.2f}', 
                        'duration': f'{duration:.2f}s'
                    })
    
                pbar.update(1)
            if args.algo=='EWC':
                model.new_consolidation(X_train[task],Y_train[task],head=task)
            if args.algo=='SI':
                TR.model.calculate_importance()
                TR.optimizer = torch.optim.Adam(TR.model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #Reset the optimizer as in SI 

    torch.save(model.state_dict(), args.result_dir+'/model.pth')
    np.savetxt(args.result_dir+'/acc_test',Acc_test)
    Acc_test=np.array(Acc_test)
    plt.figure()
    for i in range(11):
        plt.plot(Acc_test[:,i])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0.1,1])
    plt.savefig(args.result_dir+'/main_result.svg') 

# USED FOR FIGURE 5      
if args.learning_scenario == 'Task incremental' and args.dataset=='CIFAR110' and args.boundary=="Unclear":
    Acc_train=[]
    Acc_test=[]
    numberOfiterations=args.seasons*args.train_epochs_A + args.seasons*args.train_epochs_B*(args.num_heads-1)
    with tqdm(total=numberOfiterations, desc='Training Progress', unit='epoch') as pbar:
        for season in range(args.seasons):
            for task in range(args.num_heads):
                if task ==0:
                    train_epochs = args.train_epochs_A
                else:
                    train_epochs = args.train_epochs_B
                if args.algo=='SI':
                    TR.model.create_si_params()
                for epoch in range(train_epochs):
                    size = len(X_train[task])//args.seasons
                    if season<args.seasons-1:
                        order = np.random.permutation(size)
                        X=X_train[task][season*size:(season+1)*size][order]
                        Y=Y_train[task][season*size:(season+1)*size][order]
                    else :
                        order = np.random.permutation(len(X_train[task])-season*size)
                        X=X_train[task][season*size:][order]
                        Y=Y_train[task][season*size:][order]

                    start_time = time.time()
                    loss=TR.train_task_inc(X,Y,head=task, batch_size=args.batch_size, samples_train=args.samples_train)
                    if (epoch+1)%2==0:
                        acc_test=[]
                        for i in range(args.num_heads):
                            acc_test.append(TR.eval_task_inc(X_test[i],Y_test[i],head=i, batch_size=args.batch_size_inf, samples_inf=args.samples_inf))
                        
                        Acc_test.append(acc_test)
                        end_time = time.time()
                        duration = end_time - start_time
                        duration = round(duration,2)
                        pbar.set_postfix({
                            'acc_test': f'{acc_test[task]:.2f}', 
                            'duration': f'{duration:.2f}s'
                        })
                    pbar.update(1)
                if args.algo=='EWC' :
                    TR.model.new_consolidation(X,Y,head=task)
                if args.algo=='SI' :
                    TR.model.calculate_importance()
                    TR.optimizer = torch.optim.Adam(TR.model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #Reset the optimizer as in SI 

    torch.save(model.state_dict(), args.result_dir+'/model.pth')
    np.savetxt(args.result_dir+'/acc_test',Acc_test)
    Acc_test=np.array(Acc_test)
    plt.figure()
    for i in range(11):
        plt.plot(Acc_test[:,i])
    plt.xlabel('Epochs')
    plt.title(Acc_test[-1].mean())
    plt.ylabel('Accuracy')
    plt.ylim([0.1,1])
    plt.savefig(args.result_dir+'/main_result.svg') 

    
### THIS IS THE CODE FOR FIGURE 2 IN THE MAIN PAPER        
if args.learning_scenario == 'Domain incremental' and args.dataset=='ANIMALS':  
    for m in range(args.moy_over):
        ACC=[]
        # Define the right model and optimizer for the experiment
        model,optimizer = utils.select_model_and_optim(vars(args))
        DL = DataLoader(vars(args))
        X_train=DL.X_train
        X_test=DL.X_test
        Y_train=DL.Y_train
        Y_test=DL.Y_test

        # Define the right Trainer
        TR = Trainer(args.learning_scenario, args.algo, optimizer, model, clamp_grad=args.clamp_grad)
        with tqdm(total=4*args.train_epochs-1, desc='Training Progress', unit='mod') as pbar:
             start_time = time.time()
             for mod in range(4):        
                 for epoch in range(args.train_epochs):
                     order=np.random.permutation(len(X_train[mod]))
            
                     loss = TR.train(X_train[mod], Y_train[mod], batch_size=args.batch_size, samples_train=args.samples_train)
                     acc1 = TR.Eval(X_test[0], Y_test[0], batch_size=args.batch_size_inf, samples_inf=args.samples_inf)
                     acc2 = TR.Eval(X_test[1], Y_test[1], batch_size=args.batch_size_inf, samples_inf=args.samples_inf)
                     acc3 = TR.Eval(X_test[2], Y_test[2], batch_size=args.batch_size_inf, samples_inf=args.samples_inf)
                     acc4 = TR.Eval(X_test[3], Y_test[3], batch_size=args.batch_size_inf, samples_inf=args.samples_inf)
                     # acc5 = TR.Eval(X_test[4], Y_test[4], batch_size=args.batch_size_inf, samples_inf=args.samples_inf)
                     ACC.append([acc1,acc2,acc3,acc4])
                     if args.algo!='DET':
                         sigma = model.layers[0].weight_sigma.mean().detach().cpu().numpy()
                     else : 
                         sigma=0
                     end_time = time.time()
                     duration = end_time - start_time
                     duration = round(duration,2)
                     pbar.set_postfix({
                         'loss': f'{loss:.3f}',
                         'acc1': f'{acc1:.3f}',
                         'acc2': f'{acc2:.3f}',
                         'acc3': f'{acc3:.3f}',
                         'acc4': f'{acc4:.3f}',
                         'duration': f'{duration:.2f}s'
                     })
                 
                     pbar.update(1)
                 
            
        ACC=np.array(ACC)
        np.savetxt(args.result_dir+'/accuracy'+str(m), ACC) 
        if args.algo!='DET':
            predictive, aleatoric, epistemic = utils.uncertainties(model, torch.concat(X_test,axis=0), batch_size=args.batch_size_inf, samples_inf=100)
            predictive_un, aleatoric_un, epistemic_un = utils.uncertainties(model, DL.X_ood, batch_size=args.batch_size_inf, samples_inf=100)
            np.savetxt(args.result_dir+'/epistemic_un'+str(m),epistemic_un.detach().cpu().numpy())
            np.savetxt(args.result_dir+'/epistemic'+str(m),epistemic.detach().cpu().numpy())
        else: 
            aleatoric = utils.uncertainties_det(model, torch.concat(X_test,axis=0), batch_size=args.batch_size_inf)
            aleatoric_un = utils.uncertainties_det(model, DL.X_ood, batch_size=args.batch_size_inf,)
            np.savetxt(args.result_dir+'/aleatoric_un'+str(m),aleatoric_un.detach().cpu().numpy())
            np.savetxt(args.result_dir+'/aleatoric'+str(m),aleatoric.detach().cpu().numpy())
           
            
           
# IF YOU WANT TO SEE SOME PLOTS BEFORE THE END OF THE SIMULATION            
def hess(sigma,sigma_prior,N):
    NH= (1/sigma**2)
    return NH/(N)
                
# THIS IS THE CODE FOR THE SUPPLEMENTARY FIGURE ON HESSIAN APPROXIMATION
if args.hessian_approx != 'None':     
    for m in range(args.moy_over):
        it = 0  
        model,optimizer = utils.select_model_and_optim(vars(args))
        DL = DataLoader(vars(args))
        X_train=DL.X_train
        X_test=DL.X_test
        Y_train=DL.Y_train
        Y_test=DL.Y_test
        TR = Trainer(args.learning_scenario, args.algo, optimizer, model, clamp_grad=args.clamp_grad)
        while it < args.L[-1]:
            order = np.random.permutation(len(X_train))
            X_train=X_train[order]
            Y_train=Y_train[order]
            start_time = time.time()
            
            if args.hessian_approx == "MetaBayes": 
                it = TR.train_bayes_approx_hessian(X_train, Y_train, it, args.L, batch_size=args.batch_size, samples_train=args.samples_train)
                for l in args.L:
                    if it==l:
                        print("We compute the hessian and its approx for the iteration", str(it))
                        sigma=model.layers[-1].weight.sigma.cpu().detach().numpy().flatten()
                        np.savetxt(args.result_dir+'/sigma'+str(it)+str(m), sigma)
                        hessian = utils.hessian_mu(model, X_train, Y_train, batch_size=args.batch_size_inf, samples_train=args.samples_inf)    
                        np.savetxt(args.result_dir+'/hessian'+str(it)+str(m), hessian)
                        # plt.scatter(hessian.flatten(), hess(sigma,args.sigma_prior,args.N), label='Task 3', color='C3')
                        # plt.xlabel('Hessian')
                        # plt.ylabel('Approx')
                        # plt.title('iteration' +str(l))
                        # plt.savefig(args.result_dir+'/'+str(l)+'main_result.svg')
                        # plt.show() 
                        
                        
            if args.hessian_approx == "Fisher": 
                it = TR.train_det_approx_hessian(X_train, Y_train, it, args.L, batch_size=args.batch_size)
                for l in args.L:
                    if it==l:
                        print("We compute the hessian and its approx for the iteration", str(it))
                        #keep small batch size for fisher...
                        fisher = utils.fisher(model, X_train, Y_train, args.batch_size)
                        hessian = utils.hessian_det(model, X_train, Y_train, args.batch_size_inf)    
                        np.savetxt(args.result_dir+'/hessian'+str(it)+str(m), hessian)
                        np.savetxt(args.result_dir+'/fisher'+str(it)+str(m), fisher)
                        # plt.scatter(hessian.flatten(), fisher.flatten(), label='Task 3', color='C3')
                        # plt.xlabel('Hessian')
                        # plt.ylabel('Approx')
                        # plt.title('iteration' +str(l))
                        # plt.savefig(args.result_dir+'/'+str(l)+'main_result.svg')
                        # plt.show()
                        
            if args.hessian_approx == "SI": 
                it = TR.train_det_synaptic_intelligence(X_train, Y_train, it, args.L, batch_size=args.batch_size)
                for l in args.L:
                    if it==l:
                        print("We compute the hessian and its approx for the iteration", str(it))
                        importance=model.calculate_importance()
                        hessian = utils.hessian_det(model, X_train, Y_train, args.batch_size_inf)    
                        np.savetxt(args.result_dir+'/hessian'+str(it)+str(m), hessian)
                        np.savetxt(args.result_dir+'/importance'+str(it)+str(m), importance)
                        # plt.scatter(hessian.flatten(), importance.flatten(), label='Task 3', color='C3')
                        # plt.xlabel('Hessian')
                        # plt.ylabel('Approx')
                        # plt.title('iteration' +str(it)+str(m))
                        # plt.savefig(args.result_dir+'/'+str(it)+str(m)+'main_result.svg')
                        # plt.show()
                   

        
               
  

# The rest of your main.py code
# ...


"""
To execute multiple Python scripts sequentially from the terminal, you can use a shell script or simply chain the commands using semicolons (;). Here’s how you can do it:
    

Create a shell script (run_simulations.sh) and include your commands in it. This way, you can easily modify and re-run your simulations without typing all commands again.

Create a file named run_simulations.sh:
        
        touch run_simulations.sh

Example of run_simulations.sh: 
    
    #!/bin/bash
    
    # Get the current timestamp for unique identification
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Create the result directory once
    result_directory="results/RESULT_${timestamp}"
    mkdir -p "$result_directory"
    
    # Execute the Python scripts with specified arguments file names
    python3 -u main.py --simu='continual_permutted_mnist' --optimizer='BGD' --result_dir="$result_directory" --argfile="arguments_simu1.txt"
    python3 -u main.py --simu='mnist_uncertainty' --optimizer='BGD' --result_dir="$result_directory" --argfile="arguments_simu2.txt"
    # Add more commands as needed, ensuring each --argfile is unique
    

Change the permissions of the script to make it executable using the chmod command.

    chmod +x run_simulations.sh

Run your shell script in the terminal:
    ./run_simulations.sh
    

"""

