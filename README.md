# Local-Global MCMC kernels: the bost of both worlds

This repository contains Python code to reproduce experiments from [**Local-Global MCMC kernels: the bost of both worlds**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/21c86d5b10cdc28664ccdadf0a29065a-Abstract-Conference.html) (NeurIPS'22).


- [Local-Global MCMC kernels: the bost of both worlds](#local-global-mcmc-kernels-the-bost-of-both-worlds)
  - [Algorithms](#algorithms)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Experiments with synthetic distributions:](#experiments-with-synthetic-distributions)
    - [Experiments with GANs on MNIST dataset](#experiments-with-gans-on-mnist-dataset)
    - [Experiments with GANs on CIFAR10 dataset](#experiments-with-gans-on-cifar10-dataset)
  - [Results](#results)
  - [Citation](#citation)

<img src="./imgs/gaussian_mixture.png" alt="i-SIR" width="900"/>


## Algorithms 
**i-SIR:**

<img src="./algs/isir.png" alt="i-SIR" width="600"/>

**Ex2MCMC:**

<img src="./algs/ex2.png" alt="Ex2MCMC" width="600"/>

**FlEx2MCMC:**

<img src="./algs/flex.png" alt="FlEx2MCMC" width="600"/>

## Setup

Create environment:

```bash
conda create -n ex2mcmc python=3.8
conda activate ex2mcmc
```

Install poetry (if absent):
```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.create false
```

Install the project:
```bash
poetry install
```

Download checkpoints:

| GAN   |     Steps     |  Path, G |  Path, D |
|:----------|:-------------:|:------:|:------:|
| DCGAN NS  | 100k      |   [TBD]() |   [TBD]() |
| SNGAN NS  | 100k      |   [TBD]() |   [TBD]() |
| WGAN GP   | --        |   [TBD]() |   [TBD]() |

## Usage

 ### Experiments with synthetic distributions:
  
| Experiment | Path | Colab |
|:----------|:-------|:-----:|
| Toyish Gaussian   |     ```exp_synthetic/toyish_gaussian.ipynb``` | [TBD]() |
| Gaussian mixture  |     ```exp_synthetic/gaussian_mixture.ipynb``` | [TBD]() |
| Banana-shaped distribution   |     ```exp_synthetic/banana.ipynb``` | [TBD]() |
| Neal's funnel distribution   |     ```exp_synthetic/funnel.ipynb``` | [TBD]() |
| FlEx for banana-shaped distribution   |     ```exp_synthetic/flex_banana.ipynb``` | [TBD]() |
| FlEx for Neal's funnel distribution   |     ```exp_synthetic/flex_funnel.ipynb``` | [TBD]() |

 ### Experiments with GANs on MNIST dataset
 
 ```mnist_experiments```

 ### Experiments with GANs on CIFAR10 dataset

```cifar10_experiments```

## Results

## Citation

```
@article{samsonov2022local,
  title={Local-Global MCMC kernels: the best of both worlds},
  author={Samsonov, Sergey and Lagutin, Evgeny and Gabri{\'e}, Marylou and Durmus, Alain and Naumov, Alexey and Moulines, Eric},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={5178--5193},
  year={2022}
}
```



