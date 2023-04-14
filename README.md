# Local-Global MCMC kernels: the bost of both worlds

This repository contains Python code to reproduce experiments from [**Local-Global MCMC kernels: the bost of both worlds**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/21c86d5b10cdc28664ccdadf0a29065a-Abstract-Conference.html) (NeurIPS'22).


- [Local-Global MCMC kernels: the bost of both worlds](#local-global-mcmc-kernels-the-bost-of-both-worlds)
  - [Algorithms](#algorithms)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Experiments with synthetic distributions:](#experiments-with-synthetic-distributions)
    - [Experiments with GANs on MNIST dataset](#experiments-with-gans-on-mnist-dataset)
    - [Experiments with GANs on CIFAR10 dataset](#experiments-with-gans-on-cifar10-dataset)
    - [Experiments with GANs on CIFAR10 and FID computation](#experiments-with-gans-on-cifar10-and-fid-computation)
  - [Results](#results)
  - [Citation](#citation)

<img src="./imgs/gaussian_mixture.png" alt="i-SIR" width="900"/>



<img src="./imgs/fid_flex.png" alt="FID" width="250"/> <img src="./imgs/is_flex.png" alt="Inception Score" width="260"/> <img src="./imgs/energy_flex.png" alt="Energy" width="270"/> 


## Algorithms 
<!-- **i-SIR:**

<img src="./algs/isir.png" alt="i-SIR" width="600"/> -->

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

CIFAR10:

| GAN   |     Steps     |  Path, G |  Path, D |
|:----------|:-------------:|:------:|:------:|
| DCGAN NS  | 100k      |   [netG_100000_steps.pth](https://drive.google.com/file/d/1gv8_qr_xa8hJzdJpBXiKr8v922EqcE-E/view?usp=share_link) |   [netD_100000_steps.pth](https://drive.google.com/file/d/1u1sPUmlvyhcbNDX2DVsR-mGOzqQ6U8sh/view?usp=share_link) |
| SNGAN, Hinge  | 100k      |   [netG.pth](https://drive.google.com/file/d/118zC_iEkN27jGLVNmDuQpMeyw7BKOUra/view?usp=share_link) |   [netD.pth](https://drive.google.com/file/d/1xU5FV59TLhAlkFubJGmJVS87HnZZ2xHT/view?usp=share_link) |

MNIST:

| GAN      |  Path |
|:----------|:-------------:|
| Vanilla  |   [vanilla_gan.pth](https://drive.google.com/file/d/1xa1v4hPQQdU2RkhjMn5sFZCITxTJ5Dhj/view?usp=share_link) |
| WGAN CP  |   [wgan.pth](https://drive.google.com/file/d/17nQJnfs2_T6kyahnkW3fu8AVY54kmRmw/view?usp=share_link) |

You also can run script to download checkpoints:

```bash
chmod +x get_ckpts.sh
./get_ckpts.sh
```

Download statistics for FID cimputation for CIFAR10 dataset:

```bash
gdown 1jjgB_iuvmoVAXPRvVTI_hBfuIz7mQgOg -O stats/fid_stats_cifar10.npz
```

<!-- | WGAN GP   | --        |   [TBD]() |   [TBD]() | -->

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
 
 ```mnist_experiments/JSGAN_samples.ipynb```

 ```mnist_experiments/WGAN_samples.ipynb```

 ### Experiments with GANs on CIFAR10 dataset

```cifar10_experiments/DCGAN_samples.ipynb```

```cifar10_experiments/SNGAN_samples.ipynb```

### Experiments with GANs on CIFAR10 and FID computation

```bash
python run.py configs/mcmc_configs/{ula/mala/isir/ex2mcmc/flex2mcmc}.yml configs/mmc_dcgan.yml
```

## Results

| GAN | MCMC | steps | Inception Score | FID  |
|:----|:-----|:------|:---------------:|:----:|
|DCGAN| none | 0     |                 |      |
|DCGAN| i-SIR  | 1k     |                 |      |
|DCGAN| MALA  | 1k     |                 |      |
|DCGAN| Ex2MCMC  | 1k   |                 |      |
|DCGAN| FlEx2MCMC  | 1k |                 |      |


Generation trajectories for DCGAN, top to bottom: i-SIR, MALA, Ex2MCMC, FlEx2MCMC:

<img src="./imgs/cifar10_dcgan_gen.png" alt="CIFAR10 generations" width="600"/> 

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




