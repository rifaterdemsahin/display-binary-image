import os
import io
import base64
import torch
import requests
from diffusers import StableDiffusionXLPipeline
import runpod
import sys
import traceback
from datetime import datetime

# ===== SETUP LOGGING =====
LOG_FILE = "/workspace/worker.log"

def log_to_file(msg):
    """Write to both stdout AND file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg, flush=True)  # Stdout
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_msg + "\n")
            f.flush()
    except Exception as e:
        print(f"Failed to write to log: {e}")

# ===== ENSURE DIRECTORIES EXIST =====
os.makedirs("/workspace/models", exist_ok=True)
os.makedirs("/workspace/cache/huggingface", exist_ok=True)

log_to_file("✓ Directories created")

pipe = None
current_model_url = None

def log_startup():
    """Log startup information"""
    log_to_file("=" * 70)
    log_to_file("🚀 SDXL WORKER STARTUP")
    log_to_file("=" * 70)
    
    # System info
    log_to_file(f"Python: {sys.version.split()[0]}")
    log_to_file(f"PyTorch: {torch.__version__}")
    
    # CUDA info
    cuda_available = torch.cuda.is_available()
    log_to_file(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)
            log_to_file(f"GPU: {device_name}")
            log_to_file(f"Memory: {total_memory:.1f}GB")
            log_to_file(f"CUDA Capability: {device_props.major}.{device_props.minor}")
        except Exception as e:
            log_to_file(f"ERROR reading GPU info: {str(e)}")
    else:
        log_to_file("WARNING: CUDA not available!")
    
    # File system
    log_to_file(f"Models dir: /workspace/models (exists: {os.path.exists('/workspace/models')})")
    log_to_file(f"Cache dir: /workspace/cache/huggingface (exists: {os.path.exists('/workspace/cache/huggingface')})")
    
    log_to_file("=" * 70)
    log_to_file("✓ WORKER READY FOR REQUESTS")
    log_to_file("=" * 70)

def download_model(url, target_path):
    """Download model from URL to persistent volume storage"""
    log_to_file(f"[DOWNLOAD] Checking model: {target_path}")
    
    if os.path.exists(target_path):
        size_gb = os.path.getsize(target_path) / (1024**3)
        log_to_file(f"[DOWNLOAD] ✓ Model cached: {size_gb:.2f}GB")
        return

    log_to_file(f"[DOWNLOAD] Starting download from {url}")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    try:
        log_to_file(f"[DOWNLOAD] Sending request (600s timeout)...")
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        log_to_file(f"[DOWNLOAD] Response received, status: {response.status_code}")
        
        total_size = int(response.headers.get('content-length', 0))
        log_to_file(f"[DOWNLOAD] Total size: {total_size / (1024**3):.2f}GB")
        
        downloaded = 0
        chunk_count = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    chunk_count += 1
                    
                    # Log progress every 100 chunks (~800KB)
                    if chunk_count % 100 == 0:
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            mb_done = downloaded / (1024**2)
                            log_to_file(f"[DOWNLOAD] {percent:.1f}% ({mb_done:.0f}MB)")
        
        size_gb = os.path.getsize(target_path) / (1024**3)
        log_to_file(f"[DOWNLOAD] ✓ Complete: {size_gb:.2f}GB in {chunk_count} chunks")
        
    except requests.exceptions.Timeout as e:
        log_to_file(f"[DOWNLOAD] ✗ TIMEOUT: {str(e)}")
        if os.path.exists(target_path):
            os.remove(target_path)
        raise RuntimeError(f"Download timeout - URL may be invalid or too slow")
    except Exception as e:
        log_to_file(f"[DOWNLOAD] ✗ ERROR: {str(e)}")
        log_to_file(f"[DOWNLOAD] Traceback:\n{traceback.format_exc()}")
        if os.path.exists(target_path):
            os.remove(target_path)
        raise RuntimeError(f"Download failed: {str(e)}")

def unload_model():
    """Safely unload current model from VRAM"""
    global pipe
    if pipe is not None:
        try:
            log_to_file("[MODEL] Unloading previous model...")
            del pipe
            torch.cuda.empty_cache()
            log_to_file("[MODEL] ✓ Model unloaded")
        except Exception as e:
            log_to_file(f"[MODEL] WARNING: Failed to unload: {str(e)}")
    pipe = None

def get_pipeline(model_url):
    """Load or retrieve cached SDXL pipeline"""
    global pipe, current_model_url
    
    log_to_file(f"[PIPELINE] Loading pipeline for URL: {model_url}")
    
    # Extract filename from URL
    model_filename = model_url.split("/")[-1].split("?")[0]
    if not model_filename.endswith(".safetensors"):
        model_filename = "custom_model.safetensors"
    
    log_to_file(f"[PIPELINE] Filename: {model_filename}")
    model_path = os.path.join("/workspace/models", model_filename)

    # Download if not cached
    try:
        download_model(model_url, model_path)
    except Exception as e:
        log_to_file(f"[PIPELINE] ✗ Download failed: {str(e)}")
        raise

    # Load new model only if URL changed or first load
    if pipe is None or current_model_url != model_url:
        log_to_file(f"[PIPELINE] Creating new pipeline")
        
        # Unload previous model to free VRAM
        if current_model_url != model_url and pipe is not None:
            unload_model()
        
        try:
            log_to_file(f"[PIPELINE] Loading from_single_file: {model_path}")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            log_to_file(f"[PIPELINE] ✓ Pipeline created")
            
            # CRITICAL: Strip all safety filters
            log_to_file(f"[PIPELINE] Removing safety filters")
            pipe.safety_checker = None
            pipe.feature_extractor = None
            pipe.requires_safety_checker = False
            log_to_file(f"[PIPELINE] ✓ Safety filters removed")
            
            # Move to GPU
            log_to_file(f"[PIPELINE] Moving to CUDA")
            pipe.to("cuda")
            log_to_file(f"[PIPELINE] ✓ Model on CUDA")
            
            current_model_url = model_url
            log_to_file(f"[PIPELINE] ✓ Pipeline ready")
            
        except Exception as e:
            log_to_file(f"[PIPELINE] ✗ FAILED: {str(e)}")
            log_to_file(f"[PIPELINE] Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to load model pipeline: {str(e)}")
    else:
        log_to_file(f"[PIPELINE] Using cached pipeline")
        
    return pipe

def handler(event):
    """Serverless handler for SDXL image generation"""
    log_to_file(f"[HANDLER] Request received")
    
    try:
        input_data = event.get("input", {})
        log_to_file(f"[HANDLER] Input keys: {list(input_data.keys())}")
        
        # Required inputs
        prompt = input_data.get("prompt", "").strip()
        model_url = input_data.get("model_url", "").strip()
        
        # Optional inputs with sensible defaults
        negative_prompt = input_data.get("negative_prompt", "")
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        num_inference_steps = input_data.get("num_inference_steps", 30)
        guidance_scale = input_data.get("guidance_scale", 7.0)
        
        log_to_file(f"[HANDLER] Prompt: {prompt[:60]}...")
        log_to_file(f"[HANDLER] Model URL: {model_url[:60]}...")
        log_to_file(f"[HANDLER] Config: {width}x{height}, {num_inference_steps} steps, guidance {guidance_scale}")
        
        # Input validation
        if not prompt:
            log_to_file(f"[HANDLER] ✗ Missing prompt")
            return {"error": "prompt is required", "status": "failed"}
        if not model_url:
            log_to_file(f"[HANDLER] ✗ Missing model_url")
            return {"error": "model_url is required", "status": "failed"}
        
        # Validate dimensions
        if not (512 <= width <= 2048 and 512 <= height <= 2048):
            log_to_file(f"[HANDLER] ✗ Invalid dimensions: {width}x{height}")
            return {"error": "width and height must be between 512 and 2048", "status": "failed"}
        
        # Validate inference steps
        if not (1 <= num_inference_steps <= 100):
            log_to_file(f"[HANDLER] ✗ Invalid steps: {num_inference_steps}")
            return {"error": "num_inference_steps must be between 1 and 100", "status": "failed"}

        # Get pipeline (load or use cached)
        log_to_file(f"[HANDLER] Loading pipeline")
        try:
            local_pipe = get_pipeline(model_url)
        except Exception as e:
            log_to_file(f"[HANDLER] ✗ Pipeline load failed: {str(e)}")
            return {"error": f"Pipeline load failed: {str(e)}", "status": "failed"}

        log_to_file(f"[HANDLER] Generating image")
        try:
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
            log_to_file(f"[HANDLER] ✓ Image generated")
        except Exception as e:
            log_to_file(f"[HANDLER] ✗ Generation failed: {str(e)}")
            log_to_file(f"[HANDLER] Traceback:\n{traceback.format_exc()}")
            return {"error": f"Generation failed: {str(e)}", "status": "failed"}

        # Encode image to base64
        log_to_file(f"[HANDLER] Encoding image to base64")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        log_to_file(f"[HANDLER] ✓ Base64 encoded ({len(img_str)} bytes)")

        log_to_file(f"[HANDLER] ✓ SUCCESS")
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

    except Exception as e:
        log_to_file(f"[HANDLER] ✗ UNHANDLED ERROR: {str(e)}")
        log_to_file(f"[HANDLER] Traceback:\n{traceback.format_exc()}")
        return {"error": f"Unexpected error: {str(e)}", "status": "failed"}

# ===== STARTUP =====
try:
    log_to_file("=" * 70)
    log_to_file("WORKER INITIALIZING")
    log_to_file("=" * 70)
    log_startup()
    
    log_to_file("[STARTUP] Starting RunPod serverless handler...")
    if __name__ == "__main__":
        runpod.serverless.start({"handler": handler})
    
except Exception as e:
    log_to_file(f"[STARTUP] ✗ FATAL ERROR: {str(e)}")
    log_to_file(f"[STARTUP] Traceback:\n{traceback.format_exc()}")
    raise
