
import torch
import open_clip

from huggingface_hub import hf_hub_download


def load_remoteclip():

    model_name = "ViT-B-32"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = hf_hub_download(
        repo_id="chendelong/RemoteCLIP",
        filename="RemoteCLIP-ViT-B-32.pt",
        cache_dir="checkpoints",
        local_files_only=True
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None
    )

    tokenizer = open_clip.get_tokenizer(model_name)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    message = model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("RemoteCLIP loaded successfully!")
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Loading status:", message)

    return model, preprocess, tokenizer, device


if __name__ == "__main__":

    model, preprocess, tokenizer, device = load_remoteclip()

    print("Model is ready for inference.")
