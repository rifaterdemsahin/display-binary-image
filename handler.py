import os
import io
import base64
import torch
import requests
from diffusers import StableDiffusionXLPipeline
import runpod

# Ensure required directories exist
os.makedirs("/workspace/models", exist_ok=True)
os.makedirs("/workspace/cache/huggingface", exist_ok=True)

pipe = None
current_model_url = None

# ✅ FIX #1: Add startup logging function
def log_startup():
    """Log that the handler is ready"""
    print("=" * 60)
    print("✓ SDXL Worker Initialized")
    print("=" * 60)
    print("✓ CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✓ GPU:", torch.cuda.get_device_name(0))
    print("✓ Ready for requests")
    print("=" * 60)

def download_model(url, target_path):
    """Download model from URL to persistent volume storage"""
    if os.path.exists(target_path):
        print(f"--> Model already cached at {target_path}")
        return

    print(f"--> Downloading model from {url}...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    try:
        # ✅ FIX #2: Increase timeout from 300 to 600 seconds
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"--> Download progress: {percent:.1f}%", end='\r')
        
        print(f"\n--> Download complete: {total_size / (1024**3):.2f}GB")
    except requests.exceptions.RequestException as e:
        # Clean up partial download
        if os.path.exists(target_path):
            os.remove(target_path)
        raise RuntimeError(f"Failed to download model: {str(e)}")

def unload_model():
    """Safely unload current model from VRAM"""
    global pipe
    if pipe is not None:
        try:
            del pipe
            torch.cuda.empty_cache()
            print("--> Previous model unloaded from VRAM")
        except Exception as e:
            print(f"--> Warning unloading model: {str(e)}")
    pipe = None

def get_pipeline(model_url):
    """Load or retrieve cached SDXL pipeline"""
    global pipe, current_model_url
    
    # Extract filename from URL
    model_filename = model_url.split("/")[-1].split("?")[0]
    if not model_filename.endswith(".safetensors"):
        model_filename = "custom_model.safetensors"
        
    model_path = os.path.join("/workspace/models", model_filename)

    # Download if not cached
    download_model(model_url, model_path)

    # Load new model only if URL changed or first load
    if pipe is None or current_model_url != model_url:
        print(f"--> Loading new model: {model_filename}")
        
        # Unload previous model to free VRAM
        if current_model_url != model_url:
            unload_model()
        
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            
            # CRITICAL: Strip all safety filters
            pipe.safety_checker = None
            pipe.feature_extractor = None
            pipe.requires_safety_checker = False
            
            pipe.to("cuda")
            current_model_url = model_url
            print("--> Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model pipeline: {str(e)}")
        
    return pipe

def handler(event):
    """Serverless handler for SDXL image generation"""
    try:
        input_data = event.get("input", {})
        
        # Required inputs
        prompt = input_data.get("prompt", "").strip()
        model_url = input_data.get("model_url", "").strip()
        
        # Optional inputs with sensible defaults
        negative_prompt = input_data.get("negative_prompt", "")
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        num_inference_steps = input_data.get("num_inference_steps", 30)
        guidance_scale = input_data.get("guidance_scale", 7.0)
        
        # Input validation
        if not prompt:
            return {"error": "prompt is required", "status": "failed"}
        if not model_url:
            return {"error": "model_url (pointing to .safetensors file) is required", "status": "failed"}
        
        # Validate dimensions
        if not (512 <= width <= 2048 and 512 <= height <= 2048):
            return {"error": "width and height must be between 512 and 2048", "status": "failed"}
        
        # Validate inference steps
        if not (1 <= num_inference_steps <= 100):
            return {"error": "num_inference_steps must be between 1 and 100", "status": "failed"}

        # Get pipeline (load or use cached)
        local_pipe = get_pipeline(model_url)

        print(f"--> Generating image: {prompt[:50]}...")
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                image = local_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]

        # Encode image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "image": f"data:image/jpeg;base64,{img_str}",
            "generation_params": {
                "prompt": prompt,
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "dimensions": f"{width}x{height}"
            }
        }

    except RuntimeError as e:
        print(f"--> Runtime error: {str(e)}")
        return {"error": str(e), "status": "failed"}
    except Exception as e:
        print(f"--> Unexpected error: {str(e)}")
        return {"error": f"Generation failed: {str(e)}", "status": "failed"}

# ✅ FIX #3: Call startup logging before handler
if __name__ == "__main__":
    log_startup()
    runpod.serverless.start({"handler": handler})
