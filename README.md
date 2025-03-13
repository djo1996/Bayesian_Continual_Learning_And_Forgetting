# Bayesian Continual Learning And Forgetting


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
```
