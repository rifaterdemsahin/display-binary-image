FROM runpod/base:0.6.2-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
    diffusers==0.27.2 \
    transformers==4.38.2 \
    accelerate==0.27.2 \
    safetensors==0.4.2 \
    requests==2.31.0 \
    torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

ENV SAFETY_CHECKER=false
ENV HF_HOME=/workspace/cache/huggingface
ENV TORCH_CUDA_ARCH_LIST=89

RUN mkdir -p /workspace/models /workspace/cache/huggingface /runpod

COPY .runpod/hub.json /runpod/hub.json
COPY .runpod/tests.json /runpod/tests.json
COPY handler_debug.py /handler.py

WORKDIR /workspace

CMD ["python", "-u", "/handler.py"]
