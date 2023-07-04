

#!/bin/bash

if ! command -v glxinfo &> /dev/null
then
    echo "glxinfo command  not found! Execute \'sudo apt install mesa-utils\' to install it."
    exit
fi

vendor=`glxinfo | grep vendor | grep OpenGL | awk '{ print $4 }'`

#xhost +local:docker

# --device=/dev/video0:/dev/video0
# For non root usage:
# RUN sudo usermod -a -G video developer

if [ $vendor == "NVIDIA" ]; then
    docker run -it --rm \
        --name lfd_desktop \
        --hostname lfd_desktop \
        --device /dev/snd \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        -v `pwd`/../Orientation_learning:/home/user/Orientation_learning \
        -env="XAUTHORITY=$XAUTH" \
        --volume="$XAUTH:$XAUTH" \
        --gpus all \
        lfd_docker:latest \
        bash
else
    docker run --privileged -it --rm \
        --name lfd_desktop \
        --hostname lfd_desktop \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix \
        -v `pwd`/../Orientation_learning:/home/user/Orientation_learning \
        --device=/dev/dri:/dev/dri \
        --env="DISPLAY=$DISPLAY" \
        -e "TERM=xterm-256color" \
        --cap-add SYS_ADMIN --device /dev/fuse \
        lfd_docker:latest \
        bash
fi