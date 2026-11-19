
import json
import torch
import numpy as np

from pathlib import Path
from remoteclip.model import load_remoteclip


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = ROOT / "data" / "embeddings"


def search_images(query, top_k=5):

    embeddings = np.load(
        EMBEDDING_DIR / "image_embeddings.npy"
    )

    with open(
        EMBEDDING_DIR / "image_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    model, preprocess, tokenizer, device = load_remoteclip()

    text_tokens = tokenizer([query]).to(device)

    with torch.inference_mode():

        text_features = model.encode_text(text_tokens)

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

    query_embedding = text_features.cpu().numpy()[0]

    similarities = embeddings @ query_embedding

    top_k = min(top_k, len(metadata))

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    print("\nSearch Query:", query)
    print("-" * 50)

    results = []

    for rank, index in enumerate(ranked_indices, start=1):

        result = {
            "rank": rank,
            "filename": metadata[index]["filename"],
            "path": metadata[index]["path"],
            "similarity": float(similarities[index])
        }

        results.append(result)

        print("Rank:", rank)
        print("Image:", result["filename"])
        print("Similarity:", f"{result['similarity']:.4f}")
        print("Path:", result["path"])
        print("-" * 50)

    return results


if __name__ == "__main__":

    query = "An airport with aircraft on the runway"

    search_images(query, top_k=5)
