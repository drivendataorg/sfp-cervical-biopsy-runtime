#!/bin/bash
set -e

# load .env vars
if [ -f .env ]; then
    cat .env | while read a; do export $a; done
fi

cd runtime

# test configuration
if [ $(which nvidia-smi) ]
then
    docker build --build-arg CPU_GPU=gpu -t ai-for-earth-serengeti/inference .
    docker run --env-file .env \
           --gpus all \
           --network none \
           --mount type=bind,source=$(pwd)/inference-data,target=/inference/data,readonly \
           --mount type=bind,source=$(pwd)/submission,target=/inference/submission \
           ai-for-earth-serengeti/inference
else
    docker build --build-arg CPU_GPU=cpu -t ai-for-earth-serengeti/inference .
    docker run --env-file .env \
            --network none \
            --mount type=bind,source=$(pwd)/inference-data,target=/inference/data,readonly \
            --mount type=bind,source=$(pwd)/submission,target=/inference/submission \
            ai-for-earth-serengeti/inference
fi
