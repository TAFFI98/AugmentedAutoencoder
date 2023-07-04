FROM ubuntu:22.04
LABEL maintainer="taffi98.at@gmail.com"
LABEL version="0.1"
LABEL description="This is a docker immage for LfD thesis"
SHELL [ "/bin/bash", "--login", "-c" ]

# Set the nvidia container runtime.
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
ARG DEBIAN_FRONTEND=noninteractive
# Install some handy tools.
RUN set -x \
        && apt-get update \
        && apt-get upgrade -y \
        && apt-get install -y apt-utils \
        && apt-get install -y mesa-utils \
        && apt-get install -y iputils-ping \
        && apt-get install -y apt-transport-https ca-certificates \
        && apt-get install -y openssh-server python3-pip exuberant-ctags \
        && apt-get install -y git vim tmux nano htop sudo curl wget gnupg2 \
        && apt-get install -y bash-completion \
        && apt-get install -y python3-psycopg2 \
        && apt-get install -y libglfw3-dev libglfw3 libassimp-dev   \
        && pip3 install powerline-shell \
        && rm -rf /var/lib/apt/lists/* \
        && useradd -ms /bin/bash user \
        && echo "user:user" | chpasswd && adduser user sudo \
        && echo "user ALL=(ALL) NOPASSWD: ALL " >> /etc/sudoers

# Install miniconda
ENV HOME /home
ENV MINICONDA_VERSION latest
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    sudo ~/miniconda.sh -b -p $CONDA_DIR && \
    sudo rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash 

WORKDIR $HOME/user


# create conda environment
RUN conda create -n aae_py37_tf26 python=3.9  \
    && conda activate aae_py37_tf26 \
    && pip install --pre --upgrade PyOpenGL PyOpenGL_accelerate \
    && pip install cython \
    && pip install cyglfw3 \
    && pip install pyassimp==3.3 \
    && pip install imgaug \
    && pip install progressbar \
    && pip install tensorflow==2.12.* \
    && pip install opencv-python


ENV PROJECT_DIR $HOME/user
RUN mkdir -p $PROJECT_DIR
RUN mkdir -p $HOME/user/Orientation_learning/
COPY ./Orientation_learning/ $HOME/user/Orientation_learning/

RUN conda activate aae_py37_tf26 \ 
    && cd ~/user/Orientation_learning/AugmentedAutoencoder \
    && pip install .

# ENV AE_WORKSPACE_PATH /home/user/
# RUN mkdir $AE_WORKSPACE_PATH \
#     && cd $AE_WORKSPACE_PATH \
#     && ae_init_workspace


COPY ./entrypoint.sh /home/user/
RUN sudo chmod u+x /home/user/entrypoint.sh
ENTRYPOINT [ "/home/user/entrypoint.sh" ]
CMD sudo service ssh start && /bin/bash  
STOPSIGNAL SIGTERM

