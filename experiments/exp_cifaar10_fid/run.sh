

for alg in 'ula' 'mala' 'isir' 'ex2mcmc' 'flex2mcmc'
do
    python experiments/exp_fid/run_mmc_dcgan.py configs/mcmc_configs/${alg}.yml configs/mmc_dcgan.yml
done