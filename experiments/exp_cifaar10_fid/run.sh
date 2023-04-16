

for alg in 'ula' 'mala' 'isir' 'ex2mcmc' 'flex2mcmc'
do
    python experiments/exp_cifar10_fid/run.py configs/mcmc_configs/${alg}.yml configs/mmc_dcgan.yml
done