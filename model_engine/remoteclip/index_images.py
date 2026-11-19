
import json
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from remoteclip.model import load_remoteclip


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT_DIR = ROOT / "data" / "embeddings"

BATCH_SIZE = 8

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def index_images():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not image_paths:
        print("No images found in:", IMAGE_DIR)
        return

    model, preprocess, tokenizer, device = load_remoteclip()

    all_embeddings = []
    metadata = []

    for start in range(0, len(image_paths), BATCH_SIZE):

        batch_paths = image_paths[start:start + BATCH_SIZE]

        image_tensors = []
        valid_paths = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                image_tensors.append(preprocess(image))
                valid_paths.append(path)
            except Exception as error:
                print("Skipping:", path.name, "|", error)

        if not image_tensors:
            continue

        batch = torch.stack(image_tensors).to(device)

        with torch.inference_mode():

            features = model.encode_image(batch)

            features = features / features.norm(
                dim=-1, keepdim=True
            )

        embeddings = features.cpu().numpy()

        all_embeddings.extend(embeddings)

        for path in valid_paths:
            metadata.append({
                "filename": path.name,
                "path": str(path.relative_to(ROOT))
            })

        print("Processed:", len(metadata), "/", len(image_paths))

    if not all_embeddings:
        print("No valid images could be processed.")
        return

    embedding_array = np.array(all_embeddings, dtype=np.float32)

    np.save(
        OUTPUT_DIR / "image_embeddings.npy",
        embedding_array
    )

    with open(
        OUTPUT_DIR / "image_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nImage indexing completed!")
    print("Total images:", len(metadata))
    print("Embedding shape:", embedding_array.shape)
    print("Saved embeddings:", OUTPUT_DIR / "image_embeddings.npy")
    print("Saved metadata:", OUTPUT_DIR / "image_metadata.json")


if __name__ == "__main__":
    index_images()
