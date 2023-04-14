
mkdir -p checkpoints/CIFAR10/SNGAN_Hinge &  gdown 118zC_iEkN27jGLVNmDuQpMeyw7BKOUra -O checkpoints/CIFAR10/SNGAN_Hinge/netG.pth

mkdir -p checkpoints/CIFAR10/SNGAN_Hinge &  gdown 1xU5FV59TLhAlkFubJGmJVS87HnZZ2xHT -O checkpoints/CIFAR10/SNGAN_Hinge/netD.pth

mkdir -p checkpoints/CIFAR10/DCGAN_NS &  gdown 1gv8_qr_xa8hJzdJpBXiKr8v922EqcE-E -O checkpoints/CIFAR10/DCGAN_NS/netG_100000_steps.pth

mkdir -p checkpoints/CIFAR10/DCGAN_NS &  gdown 1u1sPUmlvyhcbNDX2DVsR-mGOzqQ6U8sh -O checkpoints/CIFAR10/DCGAN_NS/netD_100000_steps.pth


mkdir -p checkpoints/MNIST &  gdown 1xa1v4hPQQdU2RkhjMn5sFZCITxTJ5Dhj -O checkpoints/MNIST/vanilla_gan.pth

mkdir -p checkpoints/MNIST &  gdown 17nQJnfs2_T6kyahnkW3fu8AVY54kmRmw -O checkpoints/MNIST/wgan.pth
