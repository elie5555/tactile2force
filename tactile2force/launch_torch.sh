#!/bin/bash

# Define the path to the virtual environment activation script
venv_path="/home/elie/python_ws/env/.torchEnv/bin/activate"

# Check if the virtual environment activation script exists
if [ -f "$venv_path" ]; then
    # Source (activate) the virtual environment
    source "$venv_path"
    echo "Virtual environment activated."
else
    echo "Error: Virtual environment activation script not found at $venv_path."
    exit 1
fi

