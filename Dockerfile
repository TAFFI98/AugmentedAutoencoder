FROM tensorflow/tensorflow:latest-gpu
LABEL maintainer="taffi98.at@gmail.com"
LABEL version="0.1"
LABEL description="This is a docker immage for LfD thesis"
SHELL [ "/bin/bash", "--login", "-c" ]
ENV HOME /home
WORKDIR $HOME/user

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
  apt-utils \
     xauth \
       build-essential \
    mesa-utils \
    libxext6 \
    libx11-6 \
    wget \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev \
  && rm -rf /var/lib/apt/lists/*

# Set the nvidia container runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN apt-get update \
&& apt-get install -y libglfw3-dev libglfw3 libassimp-dev   



COPY env.yml /tmp/

RUN chmod +x /tmp/env.yml 

# ======== Install miniconda ========
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
RUN conda env create -f /tmp/env.yml 

RUN mkdir -p $HOME/user/Orientation_learning/
COPY ./Orientation_learning/ $HOME/user/Orientation_learning/


RUN /bin/bash -c "conda activate aae_py37_tf26 ; pip install cython ; pip install cyglfw3"

RUN /bin/bash -c " cd /home/user/Orientation_learning/AugmentedAutoencoder ; pip install . ; export AE_WORKSPACE_PATH=\"/home/user/Orientation_learning/AugmentedAutoencoder/AAE_workspace\" ; echo $AE_WORKSPACE_PATH ; export PYOPENGL_PLATFORM='egl' "





COPY ./entrypoint.sh /home/user/
RUN  chmod u+x /home/user/entrypoint.sh
ENTRYPOINT [ "/home/user/entrypoint.sh" ]

