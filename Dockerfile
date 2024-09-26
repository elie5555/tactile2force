# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid user interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the system and install Python, pip, and the virtualenv package
RUN apt-get update && apt-get install -y python3.9 python3-pip python3-venv && apt-get clean

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create a virtual environment called 'myenv' in the container
RUN python3 -m venv /opt/myenv

# Copy the requirements.txt file into the container
COPY tactile2force/requirements.txt /tmp/requirements.txt

# Activate the virtual environment and install the required Python packages
# Notice we use the source command to activate the virtual environment
RUN /opt/myenv/bin/pip install --upgrade pip && \
    /opt/myenv/bin/pip install -r /tmp/requirements.txt &&\
    /opt/myenv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint script to automatically source the virtual environment
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default to starting a bash shell
CMD ["/bin/bash"]
