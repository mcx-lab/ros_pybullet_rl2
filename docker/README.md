# Docker support for GAIL

 ## Github
Clone the repository
```
git clone -b gail https://github.com/mcx-lab/ros_pybullet_rl2.git
```

## Dockerfile
Build the Docker image from the Dockerfile

### For Nvidia GPUs
```
cd ros_pybullet_rl2
docker build -t pybulletrl2-gail:latest -f docker/Dockerfile.kinetic .
chmod a+x docker/run_nvidia.bash
docker/run_nvidia.bash
```

## Within container
Activate the conda environment
```
conda activate py3.7
```

Navigate to root/gail_ws and build the package
```
catkin_make
```
Then source it
```
source devel/setup.bash
```

Run GAIL training
```
roslaunch ros_pybullet_rl2 nav_train_gail.launch
```

## Docker
Do the following steps to get into the container subsequently once it has been created

Run this to allow for GUI once per session (i.e. after every reboot)
```
xhost +si:localuser:$USER
xhost +local:docker
```
Then start the container and enter it
```
docker start pybulletrl2-gail
docker exec -it pybulletrl2-gail bash
```

## Clean Slate
To delete the container, run
```
docker rm -f pybulletrl2-gail
```