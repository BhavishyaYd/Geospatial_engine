
import json
import time
import torch
import numpy as np

from pathlib import Path
from remoteclip.model import load_remoteclip


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = ROOT / "data" / "embeddings"


class RemoteCLIPEngine:

    def __init__(self):

        start = time.perf_counter()

        self.model, self.preprocess, self.tokenizer, self.device = (
            load_remoteclip()
        )

        embeddings = np.load(
            EMBEDDING_DIR / "image_embeddings.npy"
        )

        with open(
            EMBEDDING_DIR / "image_metadata.json",
            "r",
            encoding="utf-8"
        ) as file:
            self.metadata = json.load(file)

        if len(embeddings) != len(self.metadata):
            raise ValueError("Embedding and metadata mismatch.")

        if embeddings.ndim != 2 or embeddings.shape[1] != 512:
            raise ValueError("Invalid embedding dimensions.")

        if len(set(item["filename"] for item in self.metadata)) != len(self.metadata):
            raise ValueError("Duplicate image filenames detected.")

        norms = np.linalg.norm(embeddings, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Image embeddings are not normalized.")

        self.image_features = torch.as_tensor(
            embeddings,
            dtype=torch.float32,
            device=self.device
        )

        if self.device == "cuda":
            torch.cuda.synchronize()

        print("\nRemoteCLIP engine initialized.")
        print("Images indexed:", len(self.metadata))
        print("Embedding shape:", tuple(self.image_features.shape))
        print("Device:", self.device)
        print("Initialization time:", round(
            time.perf_counter() - start, 3
        ), "seconds")

    def search(self, query, top_k=5):

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        start = time.perf_counter()

        tokens = self.tokenizer([query]).to(self.device)

        with torch.inference_mode():

            text_features = self.model.encode_text(tokens)

            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            scores = (
                text_features @ self.image_features.T
            )[0]

        if self.device == "cuda":
            torch.cuda.synchronize()

        scores = scores.cpu().numpy()

        indices = np.argsort(scores)[::-1][:min(
            top_k, len(self.metadata)
        )]

        results = []

        for index in indices:

            results.append({
                "filename": self.metadata[index]["filename"],
                "path": self.metadata[index]["path"],
                "similarity": float(scores[index])
            })

        elapsed = time.perf_counter() - start

        print("\nQuery:", query)
        print("Search time:", round(elapsed, 4), "seconds")

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['filename']} "
                f"| {result['similarity']:.4f}"
            )

        return results


if __name__ == "__main__":

    engine = RemoteCLIPEngine()

    engine.search("Dense residential buildings")
    engine.search("Agricultural fields")
    engine.search("Roads and highways")

    if torch.cuda.is_available():
        print(
            "\nPeak GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / (1024 ** 2),
                2
            ),
            "MB"
        )
