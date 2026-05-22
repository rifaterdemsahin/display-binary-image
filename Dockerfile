# Use the official RunPod SDXL worker as the base layer
FROM ghcr.io/runpod/worker-sdxl:2.1.0

# Set the working directory inside the container
WORKDIR /

# Disable the safety checker at the system level 
ENV SAFETY_CHECKER=false
ENV HF_HOME=/workspace/cache/huggingface

# Create the standard runpod metadata directory structure
RUN mkdir -p /runpod

# Copy your configuration files into the container image
# Assumes hub.json and tests.json are in the same directory as this Dockerfile
COPY hub.json /runpod/hub.json
COPY tests.json /runpod/tests.json

# Optional: Pre-create the cache volume mounting directory 
RUN mkdir -p /workspace/cache/huggingface

# The base image already handles the CMD entrypoint to kick off the serverless handler loop,
# so we do not override it here unless customizing the handler.py logic.
