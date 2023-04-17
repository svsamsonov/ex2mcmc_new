from setuptools import setup, find_packages

setup(
    name="ex2mcmc",
    version="0.2.0",
    description="",
    author="",
    packages=find_packages(exclude=["tests*"]),
    install_requires = [
            'torch-mimicry @ git+https://github.com/kwotsin/mimicry.git@a7fda06c4aff1e6af8dc4c4a35ed6636e434c766',
            'tqdm',
            'pyro-ppl',
            'jaxlib',
            'jax',
            'easydict',
            'seaborn',
        ]
)