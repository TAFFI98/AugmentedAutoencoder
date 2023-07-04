

#!/bin/bash

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH

docker run --privileged -it --rm \
      --runtime=nvidia --gpus all \
        --name lfd_desktop \
        --hostname lfd_desktop \
        --volume=$XSOCK:$XSOCK:rw \
        --volume=$XAUTH:$XAUTH:rw \
        -v `pwd`/Orientation_learning:/home/user/Orientation_learning \
        --env="XAUTHORITY=${XAUTH}" \
        --env="DISPLAY" \
        --privileged \
        lfd_docker:latest \
        bash



XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -


