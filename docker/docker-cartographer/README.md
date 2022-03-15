# Docker support for GAIL with Cartographer ROS

 ## Github
Clone the repository
```
git clone -b gail https://github.com/mcx-lab/ros_pybullet_rl2.git
```

## Build docker image 
Build command for Nvidia GPU:

```
docker build -t cartographer-gail:latest -f docker/docker-cartographer/Dockerfile-Cartographer.kinetic .
```

## Dockerfile
Build the Docker image from the Dockerfile

### For Nvidia GPUs
```
cd ros_pybullet_rl2
docker build -t cartographer-gail:latest -f docker/docker-cartographer/Dockerfile-Cartographer.kinetic .
chmod a+x docker/docker-cartographer/run_nvidia.bash
docker/docker-cartographer/run_nvidia.bash
```

## Test
Run the playback script with existing test data
```
python3 mcx_legged_gym/legged_gym/scripts/play.py --task=a1
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
docker start mcx_legged_gym
docker exec -it mcx_legged_gym bash
```

## Clean Slate
To delete the container, run
```
docker rm -f mcx_legged_gym
```