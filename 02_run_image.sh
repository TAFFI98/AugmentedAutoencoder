XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -



# Start container
docker run \
  --rm \
  --runtime=nvidia \
  --gpus all \
  --name lfd_desktop \
  -it \
           -v /dev/video0:/dev/video0 \
           -v /dev/video1:/dev/video1 \
           -v /dev/media0:/dev/media0 \
           --volume=$XSOCK:$XSOCK:rw \
           --volume=$XAUTH:$XAUTH:rw \
           -v ./Orientation_learning:/home/user/Orientation_learning \
           --env="XAUTHORITY=${XAUTH}" \
           --env="DISPLAY" \
  --privileged \
        lfd_docker:latest \
  bash


