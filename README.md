# Bayesian Continual Learning And Forgetting

MESU is a Bayesian framework that balances learning and forgetting by leveraging synaptic uncertainty, enabling continual learning without task boundaries while mitigating catastrophic forgetting, and catastrophic remembering.

## Get Started

To set up your environment, run the following commands:

```bash
python3 -m venv mesu
source mesu/bin/activate
git clone https://github.com/djo1996/Bayesian_Continual_Learning_And_Forgetting.git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install matplotlib
pip install python-mnist
```

## Dataset Download

The CIFAR/MNIST training datasets are too large for GitHub. You can download them here:

- **[CIFAR/MNIST](https://www.dropbox.com/scl/fo/oomzzdq99ldkfyyvoylnq/ABxo-0qRKecwa4pzLjb_dpQ?rlkey=0k2r4zpnwgnratzm8m60dlb05&st=ja2716mf&dl=0)**

After downloading, place them in the `datasets/` folder before running the scripts.

## Figure Animals
This is the commands to run in order to obtain results of Figure 2.
Before running the command, ensure that you create the repository where you want to store the results. Replace "YOUR_DIRECTORY" with your desired path. (Choose a different one for each simulation)
### MESU
```bash
python3 -u main.py --learning_scenario='Domain incremental'  --dataset='ANIMALS'  --algo='MESU' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --batch_size=1 --batch_size_inf=32 --train_epochs=5 --samples_train=10 --samples_inf=100 --activation='Relu' --clamp_grad=0.0 --archi_fcnn 512 64 5 --N=5e5  --reduction='mean' --coeff_likeli=1 --random_seed=1  --c_sigma=60 --sigma_prior=0.1  --ratio_max=0.5 --c_mu=1 --clamp_sigma 1e-4 0.1 --moy_over=50
```
### SGD
```bash
python3 -u main.py --learning_scenario='Domain incremental'  --dataset='ANIMALS'  --algo='DET' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --batch_size=1 --batch_size_inf=32 --train_epochs=5 --activation='Relu'  --archi_fcnn 512 64 5  --reduction='mean' --coeff_likeli=1 --random_seed=1   --moy_over=50 --torch_optim_name='SGD' --lr=0.005
```
## Figure CIFAR

This is the commands to run in order to obtain results for clear task boundary.
This corresponds to the case where the number of splits is 1, as shown in Figure 5. This experiment was conducted to determine the optimal hyperparameters for each algorithm.

Before running the command, ensure that you create the repository where you want to store the results. Replace "YOUR_DIRECTORY" with your desired path. (Choose a different one for each simulation)

### MESU
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='MESU' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Clear" --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='sum' --random_seed=10 --num_heads=11 --num_classes=10 --moy_over=1 --samples_train=8 --samples_inf=8 --c_sigma=132 --N=1e6 --c_mu=5 --sigma_prior=1 --clamp_sigma 1e-6 1 --ratio_max=0.02

```
This is the commands to run in order to obtain results for cases where the number of splits is greater than 1—i.e., the case of an unclear boundary.
All hyperparameters must remain the same as in the previous experiment. As you can see, the only change is modifying the "boundary" argument from "Clear" to "Unclear", while selecting the desired number of seasons (number of splits).
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='MESU' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=2 --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='sum' --random_seed=10 --num_heads=11 --num_classes=10 --moy_over=1 --samples_train=8 --samples_inf=8 --c_sigma=132 --N=1e6 --c_mu=5 --sigma_prior=1 --clamp_sigma 1e-6 1 --ratio_max=0.02
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='MESU' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=4 --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='sum' --random_seed=10 --num_heads=11 --num_classes=10 --moy_over=1 --samples_train=8 --samples_inf=8 --c_sigma=132 --N=1e6 --c_mu=5 --sigma_prior=1 --clamp_sigma 1e-6 1 --ratio_max=0.02
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='MESU' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=8 --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='sum' --random_seed=10 --num_heads=11 --num_classes=10 --moy_over=1 --samples_train=8 --samples_inf=8 --c_sigma=132 --N=1e6 --c_mu=5 --sigma_prior=1 --clamp_sigma 1e-6 1 --ratio_max=0.02
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='MESU' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=16 --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='sum' --random_seed=10 --num_heads=11 --num_classes=10 --moy_over=1 --samples_train=8 --samples_inf=8 --c_sigma=132 --N=1e6 --c_mu=5 --sigma_prior=1 --clamp_sigma 1e-6 1 --ratio_max=0.02 
```

### Elastic Weight Consolidation (EWC)
```bibtex
@article{kirkpatrick2017overcoming,
  title={Overcoming catastrophic forgetting in neural networks},
  author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
  journal={Proceedings of the national academy of sciences},
  volume={114},
  number={13},
  pages={3521--3526},
  year={2017},
  publisher={National Academy of Sciences}
}
```
Clear task boundary
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='EWC' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Clear" --batch_size=200 --batch_size_inf=200 --batch_size_fisher=1 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --lambda=5 --torch_optim_name='Adam'
```
Unclear task boundary
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='EWC' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=2 --batch_size=200 --batch_size_inf=200 --batch_size_fisher=1 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --lambda=5 --torch_optim_name='Adam'
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='EWC' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=4 --batch_size=200 --batch_size_inf=200 --batch_size_fisher=1 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --lambda=5 --torch_optim_name='Adam'
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='EWC' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=8 --batch_size=200 --batch_size_inf=200 --batch_size_fisher=1 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --lambda=5 --torch_optim_name='Adam'
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='EWC' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=16 --batch_size=200 --batch_size_inf=200 --batch_size_fisher=1 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --lambda=5 --torch_optim_name='Adam'
```

### Synaptic intelligence (SI)
```bibtex
@inproceedings{zenke2017continual,
  title={Continual learning through synaptic intelligence},
  author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
  booktitle={International conference on machine learning},
  pages={3987--3995},
  year={2017},
  organization={PMLR}
}
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='SI' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Clear" --batch_size=200  --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --c_si=0.08 --torch_optim_name='Adam' 
```
 Unclear task boundary

```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='SI' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=2 --batch_size=200 - --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --c_si=0.08 --torch_optim_name='Adam' > 
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='SI' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=4 --batch_size=200 - --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --c_si=0.08 --torch_optim_name='Adam' > 
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='SI' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=8 --batch_size=200 - --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --c_si=0.08 --torch_optim_name='Adam' > 
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='SI' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=16 --batch_size=200 - --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001 --c_si=0.08 --torch_optim_name='Adam' > 
```
### Baseline
Clear task boundary

```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='DET' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Clear" --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001  --torch_optim_name='Adam'
```
 Unclear task boundary
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='DET' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=2 --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001  --torch_optim_name='Adam'
```
 ```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='DET' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=4 --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001  --torch_optim_name='Adam'
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='DET' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=8 --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001  --torch_optim_name='Adam'
```
```bash
python3 -u main.py --learning_scenario='Task incremental' --dataset="CIFAR110" --algo='DET' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --boundary="Unclear" --seasons=16 --batch_size=200 --batch_size_inf=200 --train_epochs_A=60 --train_epochs_B=60 --activation='Relu' --reduction='mean' --random_seed=10 --num_heads=11 --num_classes=10 --lr=0.001  --torch_optim_name='Adam'
```

## Supplementary Figure Hessian Approximation

This is the commands to run in order to obtain results of Supplementary Figure 1.
Before running the command, ensure that you create the repository where you want to store the results. Replace "YOUR_DIRECTORY" with your desired path. (Choose a different one for each simulation)

### MESU
```bash
python3 -u main.py --learning_scenario='Classic' --dataset='MNIST'  --algo='MESU' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --batch_size=32 --batch_size_inf=6000 --train_epochs=200 --samples_train=20 --samples_inf=20 --activation='Relu' --L 1e3 1e4 1e5 1e6  --archi_fcnn 784 50 10  --hessian_approx='MetaBayes' --input_transformation='standardize' --moy_over=3 --reduction='sum' --ratio_max=0.05  --N=1e5
```
### BGD
```bash
python3 -u main.py --learning_scenario='Classic' --dataset='MNIST'  --algo='BGD' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --batch_size=32 --batch_size_inf=6000 --train_epochs=200 --samples_train=20 --samples_inf=20 --activation='Relu' --L 1e3 1e4 1e5 1e6  --archi_fcnn 784 50 10  --hessian_approx='MetaBayes' --input_transformation='standardize' --moy_over=3 --reduction='sum' 
```

### EWC
```bash
python3 -u main.py --learning_scenario='Classic' --dataset='MNIST'  --algo='DET' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --batch_size=32 --batch_size_inf=6000 --train_epochs=200 --samples_train=3 --samples_inf=10 --activation='Relu' --L 1e3 1e4 1e5 1e6  --archi_fcnn 784 50 10  --hessian_approx='Fisher' --input_transformation='standardize' --moy_over=3 --reduction='sum' --lr=0.0001
```

### SI
```bash
python3 -u main.py --learning_scenario='Classic' --dataset='MNIST'  --algo='DET' --result_dir="YOUR_DIRECTORY" --argfile="arguments_simu.txt" --batch_size=32 --batch_size_inf=6000 --train_epochs=200 --samples_train=3 --samples_inf=10 --activation='Relu' --L 1e3 1e4 1e5 1e6  --archi_fcnn 784 50 10  --hessian_approx='SI' --input_transformation='standardize' --moy_over=3 --reduction='sum' --lr=0.0001
```



## Authors

- [Djohan BONNET](https://scholar.google.com/citations?user=1cSwOPIAAAAJ&hl=en)
- [Kellian COTTART](https://scholar.google.com/citations?hl=en&user=Akg-AH4AAAAJ)

## Citation

Please reference this work as

## License

This project is licensed under the CC-BY License - see the [LICENSE](LICENSE) file for details.
