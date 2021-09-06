# Docker support for ros_pybullet_rl2

## Build docker image

This custom image installs:
 
- ros kinetic
- all dependencies installed via apt-get
- anaconda3 and creates python3.7 env (env name=py3.7)
- sublime text (optional, line 44-48)
- terminator (optional, line 51)

Build command for nvidia gpu:

    docker build -t rl_pybullet2_gpu:latest \
        -f docker/Dockerfile-rl2-gpu . 

Build command for integrated graphics:

    docker build -t rl_pybullet2:latest \
        -f docker/Dockerfile-rl2 .

Note: 

* Build from ros_pybullet_rl2 dir so that docker can copy this package into the image.

## Make docker container 

Nvidia Gpu:

	docker run -it --privileged --net=host --ipc=host \
         --name=pybullet_rl2 \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         --runtime=nvidia \
         --gpus all \
         rl_pybullet2_gpu:latest \
         terminator

Integrated graphics:

    docker run -it --privileged --net=host --ipc=host \
         --name=pybullet_rl2 \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         rl_pybullet2:latest \
         terminator

Note: if not using terminator, replace `terminator` with `bash`

## Running container

Run,

    ./docker/run.sh pybullet_rl2

## Install pip dependencies

Couldn't install these pip stuffs within the conda env and virtualenv in dockerfile. Have to do it after creating the container. I put all the stuffs in bash script in the container.

Run,

    ./root/install.sh
    
## Remove container

Run,

	docker rm -f pybullet_rl2
