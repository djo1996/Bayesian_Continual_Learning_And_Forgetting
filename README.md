# Bayesian Continual Learning And Forgetting

MESU is a Bayesian framework that balances learning and forgetting by leveraging synaptic uncertainty, enabling continual learning without task boundaries while mitigating catastrophic forgetting, and catastrophic remembering.

## Get Started

To set up your environment, run the following commands:

```bash
python3 -m venv mesu
source mesu/bin/activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install matplotlib
pip install python-mnist
```

## Dataset Download

The CIFAR/MNIST training datasets are too large for GitHub. You can download them here:

- **[CIFAR/MNIST](https://www.dropbox.com/scl/fo/oomzzdq99ldkfyyvoylnq/ABxo-0qRKecwa4pzLjb_dpQ?rlkey=0k2r4zpnwgnratzm8m60dlb05&st=ja2716mf&dl=0)**

After downloading, place them in the `datasets/` folder before running the scripts.



## Figure CIFAR

```bash
python3 -u main.py \
    --learning_scenario='Task incremental' \
    --dataset="CIFAR110"  \
    --algo='EWC' \
    --result_dir="YOUR_DIRECTORY" \
    --argfile="arguments_simu.txt" \
    --boundary="Clear" \
    --batch_size=200 \
    --batch_size_inf=200 \
    --batch_size_fisher=1 \
    --train_epochs_A=60 \
    --train_epochs_B=60 \
    --activation='Relu'  \
    --reduction='mean' \
    --random_seed=10 \
    --num_heads=11 \
    --num_classes=10 \
    --lr=0.001 \
    --lambda=5 \
    --torch_optim_name='Adam' 
```
