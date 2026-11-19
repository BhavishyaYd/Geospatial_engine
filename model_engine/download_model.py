
from huggingface_hub import hf_hub_download
import os

model_name = "ViT-B-32"

checkpoint_path = hf_hub_download(
    repo_id="chendelong/RemoteCLIP",
    filename=f"RemoteCLIP-{model_name}.pt",
    cache_dir="checkpoints"
)

print("RemoteCLIP downloaded successfully!")
print("Checkpoint location:", checkpoint_path)
print("File size:", round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2), "MB")
