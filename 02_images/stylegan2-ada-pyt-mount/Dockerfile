# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /home

# Set default timezone for tzdata, otherwise tzdata hangs on docker build
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV DNNLIB_CACHE_DIR=/home/.cache 

RUN apt-get update
# Needed for opencv
RUN apt install -y libgl1-mesa-glx

RUN apt-get update && \
    apt install wget && \
    apt-get -y install git && \
    apt-get update --fix-missing && \ 
    apt install nano cifs-utils -y && \
    mkdir /home/.cache && \
    chmod 777 /home/.cache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -Rf /tmp/* 
    
## Latex
RUN apt update && \
    apt install texlive -y && \
    apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super -y

# bind net storage
ENV user=bra45451
ENV root_home=//fs-home.hs-regensburg.de/home/${user}/docker/
ENV root_projects=//fs-projekte.hs-regensburg.de/projekte/MB/Labore/Robotics/019_ChewRob/99_Homes/${user}/depo/
ENV mount_home ./home_bra
ENV mount_proj ./proj_depo
ENV domain=hs-regensburg.de
RUN echo mount -t cifs ${root_home} ${mount_home} -o domain=${domain},user=${user},vers=3.0 > mount.sh
RUN echo mount -t cifs ${root_projects} ${mount_proj} -o domain=${domain},user=${user},vers=3.0 >> mount.sh
RUN mkdir ${mount_home} &&  mkdir ${mount_proj}

RUN ln -s ${mount_home}/01_scripts/07_stylegan2-pyt ./07_stylegan2-pyt
RUN ln -s ${mount_proj}/docker/models/stylegan2 ./models
RUN ln -s ${mount_proj}/docker/data/einzelzahn/images ./images


RUN chmod +x ./mount.sh

RUN git clone https://github.com/dvschultz/stylegan2-ada-pytorch.git

RUN pip install --upgrade pip && \
    pip install scipy==1.3.3 requests==2.22.0 Pillow==6.2.1 h5py==2.9.0 imageio==2.9.0 \ 
    imageio-ffmpeg==0.4.3 tqdm==4.49.0 boto3==1.16.25 \
    matplotlib seaborn yapf pandas ipykernel scikit-learn opencv-python \
    pyspng==0.1.0 numpy opensimplex && \
    rm -Rf /tmp/*

RUN echo 'export PYTHONPATH="${PYTHONPATH}:/home/home_bra/01_scripts/modules"' >> ~/.bashrc && source ~/.bashrc
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/home/home_bra/01_scripts/modules/stylegan2_ada_pytorch_bra"' >> ~/.bashrc && source ~/.bashrc

# Unset TORCH_CUDA_ARCH_LIST and exec.  This makes pytorch run-time
# extension builds significantly faster as we only compile for the
# currently active GPU configuration.
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh

ENTRYPOINT ["bash","-c"]
CMD ["nvidia-smi && /entry.sh && ./mount.sh && /bin/bash"]



