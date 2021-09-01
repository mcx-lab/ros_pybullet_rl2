# Docker support for ros_pybullet_rl_

## Build docker image

This custom image installs:
 
- ros kinetic
- all dependencies installed via apt-get
- anaconda3 and creates python3.6 env (env name=py3.6)
- sublime text (optional, line 44-48)
- terminator (optional, line 51)

Build command:

	docker build -t rl_pybullet:latest \
		-f docker/Dockerfile-rl . 

Note: 

* Build from ros_pybullet_rl_  dir so that docker can copy this package into the image.
* For computers with **integrated graphics**, comment out line 11-16 of `Dockerfile-rl`.
* For computers with **nvidia gpu**, make sure [nvidia-docker 2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

## Make docker container 

Nvidia Gpu:

	docker run -it --privileged --net=host --ipc=host \
         --name=pybullet_rl \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         --runtime=nvidia \
         rl_pybullet:latest \
         terminator

Integrated graphics:

    docker run -it --privileged --net=host --ipc=host \
         --name=pybullet_rl \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         rl_pybullet:latest \
         terminator

Note: if not using terminator, replace `terminator` with `bash`

## Running container

Run,

    ./docker/run.sh pybullet_rl

## Install pip dependencies

Couldn't install these pip stuffs within the conda env and virtualenv in dockerfile. Have to do it after creating the container. I put all the stuffs in bash script in the container.

Run,

    ./root/install.sh
    
## Remove container

Run,

	docker rm -f pybullet_rl
