import os
import io
import base64
import torch
import requests
from diffusers import StableDiffusionXLPipeline
import runpod

# Global variable to cache the active pipeline in memory across warm requests
pipe = None
current_model_url = None

def download_model(url, target_path):
    """Downloads the .safetensors model checkpoint to the persistent volume if it doesn't exist."""
    if os.path.exists(target_path):
        print(f"--> Model already exists at {target_path}. Skipping download.")
        return

    print(f"--> Downloading model from {url} to {target_path}...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("--> Download complete.")

def get_pipeline(model_url):
    """Initializes or updates the SDXL pipeline while explicitly stripping the safety checker."""
    global pipe, current_model_url
    
    # Generate a deterministic filename based on the URL hash or name
    model_filename = model_url.split("/")[-1].split("?")[0]
    if not model_filename.endswith(".safetensors"):
        model_filename = "custom_model.safetensors"
        
    model_path = os.path.join("/workspace/models", model_filename)

    # Download if missing
    download_model(model_url, model_path)

    # If the model changed or pipeline isn't loaded yet, build it
    if pipe is None or current_model_url != model_url:
        print(f"--> Loading pipeline with model: {model_path}")
        
        # Load directly from single safetensors storage file
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        # CRITICAL: Strip the safety checker and feature extractor from the execution pipeline
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None
            
        pipe.to("cuda")
        current_model_url = model_url
        
    return pipe

def handler(event):
    """Processes the serverless API event data."""
    try:
        input_data = event["input"]
        
        # 1. Extract and map inputs with fallbacks matching your hub.json
        prompt = input_data.get("prompt")
        negative_prompt = input_data.get("negative_prompt", "")
        model_url = input_data.get("model_url")
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        num_inference_steps = input_data.get("num_inference_steps", 30)
        guidance_scale = input_data.get("guidance_scale", 7.0)

        if not prompt:
            return {"error": "A generation prompt is required."}
        if not model_url:
            return {"error": "A model_url pointing to an uncensored .safetensors file is required."}

        # 2. Retrieve the filter-free inference engine
        local_pipe = get_pipeline(model_url)

        # 3. Generate image (safety_checker=None inside ensures unfiltered rendering)
        print(f"--> Generating image for prompt: '{prompt}'")
        with torch.inference_mode():
            image = local_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        # 4. Convert the image buffer to a base64 payload response
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "image": f"data:image/jpeg;base64,{img_str}"
        }

    except Exception as e:
        print(f"--> Execution Error: {str(e)}")
        return {"error": str(e), "status": "failed"}

# Start the RunPod serverless worker routine loop
runpod.serverless.start({"handler": handler})
