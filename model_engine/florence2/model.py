from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_NAME = "microsoft/Florence-2-base"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "florence2"


def load_florence2():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Loading Florence-2...")
    print("Device:", device)
    print("Data type:", dtype)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(CHECKPOINT_DIR)
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        cache_dir=str(CHECKPOINT_DIR)
    )

    model = model.to(device)
    model.eval()

    print("Florence-2 loaded successfully!")
    print("Model:", MODEL_NAME)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory allocated:",
            round(torch.cuda.memory_allocated() / 1024**2, 2),
            "MB"
        )

    return model, processor, device