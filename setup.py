from setuptools import find_packages, setup


setup(
    name="ex2mcmc",
    version="0.1.0",
    description="",
    author="",
    packages=find_packages(exclude=["tests*"]),
    url="http://github.com/svsamsonov/ex2mcmc_new",
    install_requires=[
        "torch-mimicry @ git+https://github.com/kwotsin/mimicry.git",
        "tqdm",
        "pyro-ppl",
        "jaxlib",
        "jax",
        "easydict",
        "seaborn",
        "ruamel.yaml",
    ],
)
