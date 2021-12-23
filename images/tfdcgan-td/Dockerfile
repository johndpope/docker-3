FROM tensorflow/tensorflow:latest-gpu-jupyter
# ^or just latest-gpu if you don't need Jupyter

WORKDIR /tf

# Set desired Python version
ENV python_version 3.8

# Install desired Python version (the current TF image is be based on Ubuntu at the moment)
RUN apt install -y python${python_version}

# Install packages
RUN apt-get update --fix-missing
RUN apt-get -y install git
RUN apt install wget -y
RUN apt install nano cifs-utils -y

# bind net storage
ENV user=bra45451
ENV root_home=//fs-home.hs-regensburg.de/home/${user}/
ENV root_projects=//fs-projekte.hs-regensburg.de/projekte/MB/Labore/Robotics/019_ChewRob/99_Homes/${user}/depo/
ENV mount_home ./home
ENV mount_proj ./proj_depo
ENV domain=hs-regensburg.de
RUN echo mount -t cifs ${root_home} ${mount_home} -o domain=${domain},user=${user},vers=3.0 > mount.sh
RUN echo mount -t cifs ${root_projects} ${mount_proj} -o domain=${domain},user=${user},vers=3.0 >> mount.sh
RUN mkdir ${mount_home} &&  mkdir ${mount_proj}

RUN chmod +x ./mount.sh

# Set default version for root user - modified version of this solution: https://jcutrer.com/linux/upgrade-python37-ubuntu1810
RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python${python_version} 1

# Update pip: https://packaging.python.org/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date
RUN python -m pip install --upgrade pip setuptools wheel

RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/dcgan.ipynb

# By copying over requirements first, we make sure that Docker will "cache"
# our installed requirements in a dedicated FS layer rather than reinstall
# them on every build
COPY requirements.txt requirements.txt

# Install the requirements
RUN python -m pip install -r requirements.txt
# RUN python -m pip install git+https://github.com/tensorflow/docs

# Only needed for Jupyter
EXPOSE 8888

# RUN hostname -I | awk '{print$1}' > host_ip.txt 

RUN mkdir /tf/home/docker/jupyter/tfdcgan

ENV jup_fname tfdcgan

RUN echo "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root &>> /tf/home/docker/jupyter/tfdcgan/${jup_fname}.txt" > start_jup.sh
RUN chmod +x start_jup.sh

# ENTRYPOINT bash
ENTRYPOINT ["bash","-c"]
CMD ["nvidia-smi && ./mount.sh && /bin/bash"]



