from pathlib import Path
from setuptools import find_packages, setup

requirements = Path('requirements.txt').open('r').readlines()

# setup(
#     name="foo",
#     version="1.0",
#     packages=find_packages(),
#     install_requires=requirements,
#     # extras_require={"dev": [x + dev[x] if dev[x] != "*" else x for x in dev]},
# )
setup()
