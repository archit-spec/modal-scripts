import modal
import os

def download_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "nisten/Biggie-SmoLlm-0.15B-Base"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)

# Create Modal image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers>=4.31.0",
        "datasets>=2.14.0",
        "accelerate>=0.26.1",
        "bitsandbytes>=0.41.1",
        "wandb",
        "tqdm",
        "jupyter",
        "ipywidgets"
    )
    .env({
        "WANDB_API_KEY": "",
        "HF_TOKEN": "",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
    })
    .run_function(download_model)
)

app = modal.App("jupyter-training", image=image)
volume = modal.Volume.from_name("model-training-vol", create_if_missing=True)

JUPYTER_TOKEN = "your_secure_token_here"  # Change this to a secure token

@app.function(
    gpu="A10G",  # Using A10G as the cost-effective choice
    timeout=86400,  # 24 hour timeout
    volumes={"/workspace": volume},
    concurrency_limit=1,
    allow_concurrent_inputs=100
)
def run_jupyter(timeout: int = 28800):  # 8 hour default timeout
    import subprocess
    import time
    
    jupyter_port = 8888
    
    with modal.forward(jupyter_port) as tunnel:
        process = subprocess.Popen([
            "jupyter", "notebook",
            "--no-browser",
            "--allow-root",
            "--ip=0.0.0.0",
            f"--port={jupyter_port}",
            "--NotebookApp.allow_origin='*'",
            "--NotebookApp.allow_remote_access=1",
            "--notebook-dir=/workspace"
        ], env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN})
        
        print(f"\nJupyter Notebook is available at: {tunnel.url}")
        print(f"Token: {JUPYTER_TOKEN}")
        
        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                if process.poll() is not None:
                    break
                time.sleep(1)
        finally:
            process.terminate()
            process.wait()

@app.function(gpu="A10G")
def get_gpu_info():
    import torch
    import subprocess
    
    # Run nvidia-smi
    subprocess.run(["nvidia-smi"])
    
    # Print PyTorch GPU info
    print("\nPyTorch GPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

if __name__ == "__main__":
    # You can check GPU info first
    get_gpu_info.remote()
    
    # Then start Jupyter
    run_jupyter.remote()
