#!/bin/bash

# Define the image name
IMAGE_NAME="tactile2force_image"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container with the tactile2force folder mounted
docker run -it --rm -v "$(pwd)/tactile2force:/tactile2force" $IMAGE_NAME /bin/bash