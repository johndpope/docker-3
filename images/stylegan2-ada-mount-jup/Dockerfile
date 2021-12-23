# Dockerfile to run StyleGAN2 ADA - Copyright 2020 by Jeff Heaton
# added mounts by Alexander Broll
# Licensed under the MIT license
# GitHub: https://github.com/jeffheaton/docker-stylegan2-ada
# DockerHub: https://hub.docker.com/r/heatonresearch/stylegan2-ada
# 
FROM tensorflow/tensorflow:1.14.0-gpu-jupyter

# Set desired Python version
ENV python_version 3.6

WORKDIR /home

# Install desired Python version (the current TF image is be based on Ubuntu at the moment)
RUN apt update && apt install -y python3-pip
RUN apt install -y python${python_version} && apt update
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
RUN python --version


RUN python -m pip install --upgrade pip && \
    python -m pip install Jupyter && \
    python -m pip install scipy==1.3.3 requests==2.22.0 Pillow==6.2.1 h5py==2.9.0 imageio==2.9.0 \ 
    imageio-ffmpeg==0.4.2 tqdm==4.49.0 boto3==1.16.25 && \
    rm -Rf /tmp/*

ENV DNNLIB_CACHE_DIR=/home/.cache 

# needed for installing expect without prompting
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt -y install expect  && \
    apt-get -y install git && \
    mkdir /home/.cache && \
    chmod 777 /home/.cache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -Rf /tmp/* 

# switch back
ENV DEBIAN_FRONTEND=interactive

RUN apt-get update --fix-missing && \
    apt install nano cifs-utils -y

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

# fixed grayscale
RUN git clone https://github.com/robgon-art/stylegan2-ada.git 

# no fix
# RUN git clone https://github.com/NVlabs/stylegan2-ada.git

ENV jup_port=8888

EXPOSE $jup_port

ENV jup_folder $mount_home/jupyter

ENV jup_start start_jup.sh

ENV jup_pwd=$"rrru"

#RUN echo "source /etc/bash.bashrc && cat ${jup_folder}/ips.txt > ${jup_folder}/${jup_fname}.txt && jupyter notebook --port=$jup_port --notebook-dir=/ --ip 0.0.0.0 --no-browser --allow-root &>> ${jup_folder}/${jup_fname}.txt" > start_jup.sh
RUN echo "source /etc/bash.bashrc &> /dev/null && ${jup_folder}/set_jup_pwd.exp &> log_jup.txt && jupyter notebook --port=$jup_port --notebook-dir=. --ip 0.0.0.0 --no-browser --allow-root &>> log_jup.txt &" > $jup_start
RUN chmod +x $jup_start

ENTRYPOINT ["bash", "-c", "nvidia-smi && ./mount.sh  && ./start_jup.sh && /bin/bash" ]



