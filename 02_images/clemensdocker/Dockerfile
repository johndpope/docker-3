#!/bin/bash
# This is an auto generated Dockerfile for ros:ros-core
# generated from docker_images/create_ros_core_image.Dockerfile.em
#FROM ubuntu:bionic
#FROM tensorflow/tensorflow:latest-gpu-py3
#FROM tensorflow/tensorflow:devel-gpu-py3
FROM tensorflow/tensorflow:2.1.0-gpu-py3
#FROM tensorflow/tensorflow:nightly-gpu-py3

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*
	
### mount net storage ###
RUN apt-get update --fix-missing
RUN apt-get -y install git
RUN apt-get install nano cifs-utils -y


# -s: symbolic link
# -f: force (replace if exists)

## bind net storage
ENV user=poc39887
ENV root=//fs-projekte.hs-regensburg.de/projekte/
ENV folder=MB/Labore/Robotics
ENV mount=./robotics
ENV domain=hs-regensburg.de

WORKDIR /data
RUN echo mount -t cifs ${root}${folder} ${mount} -o domain=${domain},user=$user,vers=3.0  > mount.sh
RUN chmod +x ./mount.sh
RUN mkdir ${mount}

### python pip install ###
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install libsm6 libxext6 libxrender-dev
RUN pip3 install --upgrade pip
RUN pip3 install -U matplotlib
RUN pip3 install -U numpy
RUN pip3 install -U scipy
RUN pip3 install -U Pillow
RUN pip3 install -U tensorflow_hub
RUN pip3 install -U tensorflow-addons
RUN pip3 install -U opencv-python
RUN pip3 install -U modAL
#RUN pip3 install -U tensorflow-probability==0.8.0
RUN pip3 install -U tensorflow-probability
RUN pip3 install -U tqdm
RUN pip3 install -U seaborn
RUN pip3 install -U pandas
RUN pip3 install -U tensorflow_transform
RUN pip3 install -U psutil
RUN pip3 install -U albumentations

## download and build openpose

### links ###
RUN ln -s /data/robotics/004_SWoB/03_Design/17_Datenbanken /data/db
RUN ln -s /data/robotics/004_SWoB/99_Homes/poc39887/02_Design/cluster/training /data/work_dir
RUN ln -s /data/robotics/004_SWoB/99_Homes/poc39887/02_Design/cluster/training/colab/network_hai/experiments/00_iros_experiments /data/experiments

## tf_openpose
WORKDIR /data 
RUN apt-get -y install swig
RUN apt-get -y install wget
RUN git clone https://github.com/gsethi2409/tf-pose-estimation.git
WORKDIR /data/tf-pose-estimation
RUN pip3 install -r requirements.txt
WORKDIR /data/tf-pose-estimation/tf_pose/pafprocess
RUN swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
RUN pip3 install git+https://github.com/adrianc-a/tf-slim.git@remove_contrib
WORKDIR /data/tf-pose-estimation/models/graph/cmu
RUN bash download.sh

## go back to data folder
WORKDIR /data




# specifies what command to run within the container
# CMD
# entrypoint?

ENTRYPOINT bash mount.sh && cd /data && /bin/bash