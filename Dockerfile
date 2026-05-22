# Use a public, fully authorized NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python 3.11, pip, curl, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as the default python command
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch compiled for CUDA 12.1 compatibility
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Install RunPod Serverless SDK and stable diffusion requirements
RUN pip install --no-cache-dir runpod diffusers transformers accelerate safetensors requests

# Set worker environment variables to block the censorship filter
ENV SAFETY_CHECKER=false
ENV HF_HOME=/workspace/cache/huggingface

# Create required configuration directory paths
RUN mkdir -p /runpod /workspace/cache/huggingface

# Copy configuration files from the .runpod folder and the handler script into place
COPY .runpod/hub.json /runpod/hub.json
COPY .runpod/tests.json /runpod/tests.json
COPY handler.py /handler.py

# Execute the serverless runtime loop directly
CMD [ "python", "-u", "/handler.py" ]
