# Dockerfile to run StyleGAN2 ADA - Copyright 2020 by Jeff Heaton
# added streamlit and mounts by Alexander Broll
# Licensed under the MIT license
# GitHub: https://github.com/jeffheaton/docker-stylegan2-ada
# DockerHub: https://hub.docker.com/r/heatonresearch/stylegan2-ada
# 
FROM tensorflow/tensorflow:1.14.0-gpu-py3

WORKDIR /home

ENV DNNLIB_CACHE_DIR=/home/.cache 

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

RUN chmod +x ./mount.sh

RUN git clone https://github.com/robgon-art/stylegan2-ada.git

RUN pip install --upgrade pip && \
    pip install scipy==1.3.3 requests==2.22.0 Pillow==6.2.1 h5py==2.9.0 imageio==2.9.0 \ 
    imageio-ffmpeg==0.4.2 tqdm==4.49.0 boto3==1.16.25&& \
	pip install numpy matplotlib tqdm plotly && \
    pip install --upgrade protobuf && \
    rm -Rf /tmp/*


EXPOSE 8501

# Dependencies for glvnd and X11.
RUN apt-get update \
    && apt-get install -y -qq --no-install-recommends \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


ENTRYPOINT ["bash","-c"]
CMD ["nvidia-smi && ./mount.sh && /bin/bash"]



