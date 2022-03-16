xhost +si:localuser:$USER
xhost +local:docker

docker run -it \
    --device=/dev/dri \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    --env="DISPLAY=$DISPLAY" \
    --gpus all \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --runtime=nvidia \
    --network=host \
    --ipc=host \
    --name=cartographer-gail \
    cartographer-gail \
    bash
