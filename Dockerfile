# Use RunPod's official public base image built specifically for serverless
FROM runpod/base:0.6.2-cuda12.1.0

# Install the machine learning libraries needed for Stable Diffusion
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors requests

# Set worker environment variables to block the censorship filter
ENV SAFETY_CHECKER=false
ENV HF_HOME=/workspace/cache/huggingface

# Create required configuration directory paths
RUN mkdir -p /runpod /workspace/cache/huggingface

# Copy your project files into place
COPY .runpod/hub.json /runpod/hub.json
COPY .runpod/tests.json /runpod/tests.json
COPY handler.py /handler.py

# Start the serverless handler
CMD [ "python", "-u", "/handler.py" ]
