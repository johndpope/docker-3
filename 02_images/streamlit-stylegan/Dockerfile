# Dockerfile to run StyleGAN2 ADA - Copyright 2020 by Jeff Heaton
# added streamlit and mounts by Alexander Broll
# Licensed under the MIT license
# GitHub: https://github.com/jeffheaton/docker-stylegan2-ada
# DockerHub: https://hub.docker.com/r/heatonresearch/stylegan2-ada
# 
FROM tensorflow/tensorflow:1.14.0-gpu-py3

WORKDIR /home

ENV DNNLIB_CACHE_DIR=/home/.cache 

RUN apt-get update
# Needed for opencv
RUN apt install -y libgl1-mesa-glx

RUN apt install wget && \
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

RUN ln -s ${mount_home}/01_scripts/05_streamlit/01_docker_img_generator_gpu ./01_docker_img_generator_gpu

RUN chmod +x ./mount.sh

# numpy==1.16.4 added because of type warning
RUN pip install --upgrade pip && \
    pip install scipy==1.3.3 requests==2.22.0 Pillow==6.2.1 h5py==2.9.0 imageio==2.9.0 \ 
    imageio-ffmpeg==0.4.2 tqdm==4.49.0 boto3==1.16.25&& \
	pip install numpy==1.16.4 matplotlib tqdm plotly streamlit yapf seaborn pandas ipykernel scikit-learn opencv-python&& \
    pip install --upgrade protobuf && \
    rm -Rf /tmp/*




RUN echo 'export PYTHONPATH="${PYTHONPATH}:/home/home_bra/01_scripts/modules"' >> ~/.bashrc && source ~/.bashrc

EXPOSE 8501

ENTRYPOINT ["bash","-c"]
CMD ["nvidia-smi && ./mount.sh && /bin/bash"]



