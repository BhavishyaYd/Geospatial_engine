
import json
import torch
import numpy as np

from pathlib import Path
from remoteclip.model import load_remoteclip


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = ROOT / "data" / "embeddings"

QUERIES = {
    "agriculture": "Agricultural fields and cultivated farmland",
    "urban": "Dense residential buildings and urban settlements",
    "roads": "Roads and highways visible in satellite imagery",
    "vegetation": "Dense forest and green vegetation",
    "industrial": "Industrial buildings and large warehouses",
    "water": "Rivers lakes reservoirs and other water bodies",
    "airport": "An airport with aircraft on the runway"
}

TOP_K = 5


def evaluate_retrieval():

    embeddings = np.load(
        EMBEDDING_DIR / "image_embeddings.npy"
    )

    with open(
        EMBEDDING_DIR / "image_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    with open(
        EMBEDDING_DIR / "image_labels.json",
        "r",
        encoding="utf-8"
    ) as file:
        labels = json.load(file)

    filenames = [item["filename"] for item in metadata]

    if embeddings.shape[0] != len(metadata):
        raise ValueError("Embedding and metadata counts do not match.")

    missing = [name for name in filenames if name not in labels]

    if missing:
        raise ValueError(f"Missing labels for: {missing}")

    model, preprocess, tokenizer, device = load_remoteclip()

    categories = list(QUERIES.keys())
    descriptions = [QUERIES[c] for c in categories]

    text_tokens = tokenizer(descriptions).to(device)

    with torch.inference_mode():

        text_features = model.encode_text(text_tokens)

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

    image_features = torch.tensor(
        embeddings,
        dtype=text_features.dtype,
        device=device
    )

    similarity_matrix = (
        text_features @ image_features.T
    ).cpu().numpy()

    all_metrics = []

    print("\nREMOTECLIP RETRIEVAL EVALUATION")
    print("=" * 65)

    for query_index, category in enumerate(categories):

        relevant_indices = {
            i for i, filename in enumerate(filenames)
            if category in labels[filename]
        }

        if not relevant_indices:
            print(f"\nSkipping {category}: no relevant images.")
            continue

        ranked_indices = np.argsort(
            similarity_matrix[query_index]
        )[::-1]

        top_indices = ranked_indices[:TOP_K]

        relevant_found = [
            index for index in top_indices
            if index in relevant_indices
        ]

        precision = len(relevant_found) / TOP_K

        recall = len(relevant_found) / len(relevant_indices)

        first_rank = next(
            (
                rank for rank, index in enumerate(
                    ranked_indices, start=1
                )
                if index in relevant_indices
            ),
            None
        )

        reciprocal_rank = (
            1 / first_rank if first_rank is not None else 0
        )

        all_metrics.append(
            (precision, recall, reciprocal_rank)
        )

        print(f"\nCategory: {category}")
        print("Relevant images:", len(relevant_indices))
        print("Precision@5:", round(precision, 4))
        print("Recall@5:", round(recall, 4))
        print("Reciprocal Rank:", round(reciprocal_rank, 4))

        print("Top 5 retrieved images:")

        for rank, index in enumerate(top_indices, start=1):
            filename = filenames[index]
            score = similarity_matrix[query_index][index]
            is_relevant = category in labels[filename]

            print(
                f"{rank}. {filename} | "
                f"Score: {score:.4f} | "
                f"Relevant: {is_relevant}"
            )

    if all_metrics:

        averages = np.mean(all_metrics, axis=0)

        print("\n" + "=" * 65)
        print("MACRO AVERAGE RESULTS")
        print("=" * 65)

        print("Precision@5:", round(averages[0], 4))
        print("Recall@5:", round(averages[1], 4))
        print("MRR:", round(averages[2], 4))


if __name__ == "__main__":
    evaluate_retrieval()
