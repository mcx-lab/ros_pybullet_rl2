#!/bin/bash
# load params
name=$1

if [ "$1" == "" ];
then
	echo 
	echo "./run.sh [container_name]"
	echo
	exit 1
fi

# Allow X server host
export id=$(docker ps -aqf "name=${name}")
xhost -local:root
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $id`

# check if container started
if [ "`docker ps -qf "name="`" == "" ]
then
	echo "Starting previously stopped container..."
	docker start "${name}"
fi

# running container
echo "Executing into container ..."
docker exec -ti ${name} bash