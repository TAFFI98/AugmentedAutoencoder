FROM tensorflow/tensorflow:latest-gpu
LABEL maintainer="taffi <taffi98.at@gmail.com>"
SHELL [ "/bin/bash", "--login", "-c" ]
# Set the nvidia container runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Install neccessary tools
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y software-properties-common ca-certificates wget\
  build-essential xauth apt-utils unzip git vim curl \
  locate libglfw3-dev libglfw3 libassimp-dev qt5-default


ENV MINICONDA_VERSION latest
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
    
    
# ========  make non-activate conda commands available ========
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash
   

RUN  mkdir -p $HOME/user/Orientation_learning/
COPY ./Orientation_learning/ $HOME/user/Orientation_learning/

WORKDIR $HOME/user/Orientation_learning/
RUN conda env create -f env.yml 
RUN conda activate aae_py37_tf26 
RUN  conda install -c anaconda cython 
RUN conda install -c conda-forge glfw 
RUN conda install -c "conda-forge/label/cf202003" glfw 
RUN pip install cyglfw3


