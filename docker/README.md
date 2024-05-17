# Docker support for ros_pybullet_rl2

## Build docker image

This custom image installs:
 
- ROS Melodic (no longer supporting Kinetic)
- all dependencies installed via apt-get
- virtualenv and creates python3.7 env (env name=py3.7)
- sublime text
- terminator

Build command for nvidia gpu:

    docker build -t rl_pybullet2_gpu_melodic:latest \
        --build-arg UID="$(id -u)" \
        --build-arg GID="$(id -g)" \
        -f docker/Dockerfile-rl2-gpu_melodic . 

Build command for integrated graphics:

    docker build -t rl_pybullet2:latest \
        -f docker/Dockerfile-rl2 .

Note: 

* Build from ros_pybullet_rl2 dir so that docker can copy this package into the image.

* Ensure all config_${ros_distro}/ros_entrypoint_${ros_distro}.sh, bashrc, and install_dependencies.sh are executable files. 

## Make docker container 

Nvidia Gpu (supports both Melodic and Kinetic):

	docker run -it --privileged --net=host --ipc=host \
         --name=pybullet_rl2 \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         --runtime=nvidia \
         --gpus all \
         rl_pybullet2_gpu_melodic:latest \
         terminator

Integrated graphics (supports ROS Kinetic):

    docker run -it --privileged --net=host --ipc=host \
         --name=pybullet_rl2 \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         rl_pybullet2:latest \
         terminator

Note: if do not want to use terminator, replace `terminator` with `bash`

## Running container

Run,

    ./docker/run.sh pybullet_rl2

Alternatively,
```
xhost +si:localuser:$USER
xhost +local:docker
docker start pybullet_rl2
```

## Install pip dependencies

Enter the container to install the remaining pip dependencies within the virtualenv.

Run,

    . /root/install.sh
    
## Remove container

Run,

	docker rm -f pybullet_rl2
