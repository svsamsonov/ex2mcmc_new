ARG UBUNTU=20.04

FROM ubuntu:$UBUNTU

ARG PROJ="ex2mcmc"
ARG PYTHON="3.8"

RUN apt-get update && apt-get install -y \
    curl \
    vim \
    wget \
    git

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH "/root/miniconda3/bin:$PATH"

RUN conda --version
# RUN conda update conda
# ENV PATH "/root/miniconda3/bin:$PATH"
# RUN conda update anaconda
# RUN conda update --all

SHELL ["/bin/bash", "--login", "-ci"]

RUN conda init bash

RUN which pip

RUN pip install poetry
ENV PATH "/root/.poetry/bin:$PATH"
RUN poetry config virtualenvs.create false

WORKDIR /app
COPY . /app/

RUN conda create -y --name $PROJ python=$PYTHON
# activate conda environment on each RUN
RUN echo "conda activate $PROJ" >> ~/.bashrc

# RUN poetry env use 3.8
# RUN poetry env use $PROJ

RUN poetry install
RUN pip poetry add  

RUN mkdir stats
RUN gdown 1jjgB_iuvmoVAXPRvVTI_hBfuIz7mQgOg -O stats/fid_stats_cifar10.npz

RUN chmod +x ./get_ckpts.sh
RUN ./get_ckpts.sh