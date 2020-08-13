#!/bin/bash
set -e

# load .env vars
if [ -f .env ]; then
    cat .env | while read a; do export $a; done
fi

# test configuration
if [ $(which nvidia-smi) ]
then
    docker build --build-arg CPU_GPU=gpu -t sfp-cervical-biopsy/inference runtime
    docker run -it \
	   --gpus all \
           --network none \
           --mount type=bind,source=$(pwd)/inference-data,target=/inference/data,readonly \
           --mount type=bind,source=$(pwd)/submission,target=/inference/submission \
           sfp-cervical-biopsy/inference
else
    docker build --build-arg CPU_GPU=cpu -t sfp-cervical-biopsy/inference runtime
    docker run \
	   --network none \
           --mount type=bind,source=$(pwd)/inference-data,target=/inference/data,readonly \
           --mount type=bind,source=$(pwd)/submission,target=/inference/submission \
           sfp-cervical-biopsy/inference
fi
