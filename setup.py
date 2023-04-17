from setuptools import setup, find_packages

setup(
    name="ex2mcmc",
    version="0.2.0",
    description="",
    author="",
    packages=find_packages(exclude=["tests*"]),
    install_requires = [
            'torch-mimicry @ git+https://github.com/kwotsin/mimicry.git',
            'tqdm',
            'pyro-ppl',
            'jaxlib',
            'jax',
            'easydict',
            'seaborn',
        ]
)