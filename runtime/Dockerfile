FROM nvidia/cuda:11.0-base-ubuntu20.04

# non-interactive package installs by setting these up front
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=UTC
ARG DEBIAN_FRONTEND=noninteractive

ARG CPU_GPU=gpu

RUN apt-get update --fix-missing && \
    apt-get install --no-install-recommends -y \
        build-essential \
        git \
        glib2.0-dev \
        libexpat1-dev \
        libglu1-mesa \
        libgsf-1-dev \
        libjpeg-turbo8-dev \
        libtiff5-dev \
        libvips \
        libvips-dev \
        libvips-tools \
        pkg-config \
        software-properties-common \
        unzip \
        wget && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

# Limited User
RUN groupadd -g 999 appuser && \
    useradd -r -u 1000 -g appuser appuser

RUN mkdir /envs /home/appuser /inference && \
    chown -R appuser /opt /envs /home/appuser /inference

COPY ./entrypoint.sh /inference/entrypoint.sh
RUN chmod +x /inference/entrypoint.sh

USER 1000

# install miniconda as system python
COPY ./Miniconda3-py37_4.8.3-Linux-x86_64.sh /opt/miniconda.sh 
RUN /bin/bash /opt/miniconda.sh -b -p /opt/conda

ENV PATH /opt/conda/bin:$PATH
COPY ./py-${CPU_GPU}.yml /envs/

# create Python environment
RUN conda update conda && \
    conda env create -f /envs/py-${CPU_GPU}.yml && \
    conda clean -a -y

RUN conda init bash

# Execute the entrypoint.sh script inside the container when we do docker run
CMD ["/bin/bash", "/inference/entrypoint.sh"]
