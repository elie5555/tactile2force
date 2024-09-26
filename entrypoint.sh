#!/bin/bash

# Source the Python virtual environment
source /opt/myenv/bin/activate

# Pass any command given to the container to the bash shell
cd tactile2force

# Start an interactive bash shell (to ensure prompt is updated correctly)
exec /bin/bash --login && cd tactile2force
