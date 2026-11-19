
import json
import time
import torch

from pathlib import Path
from PIL import Image

from remoteclip.model import load_remoteclip


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REGION_DIR = (
    ROOT
    / "outputs"
    / "csdnet_inference"
    / "regions"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "remoteclip_temporal"
)


class RemoteCLIPTemporalComparison:

    def __init__(
        self,
        region_dir=DEFAULT_REGION_DIR,
        output_dir=DEFAULT_OUTPUT_DIR
    ):

        start = time.perf_counter()

        self.region_dir = Path(region_dir)
        self.output_dir = Path(output_dir)

        (
            self.model,
            self.preprocess,
            self.tokenizer,
            self.device
        ) = load_remoteclip()

        self.model.eval()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print("\nRemoteCLIP temporal comparison initialized.")
        print("Device:", self.device)
        print("Region directory:", self.region_dir)

        print(
            "Initialization time:",
            round(time.perf_counter() - start, 3),
            "seconds"
        )

    def encode_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.preprocess(image).unsqueeze(0).to(
            self.device
        )

        with torch.inference_mode():

            features = self.model.encode_image(image_tensor)

            features = features / features.norm(
                dim=-1,
                keepdim=True
            )

        return features

    def compare_pair(self, t1_path, t2_path, region_id):

        t1_features = self.encode_image(t1_path)
        t2_features = self.encode_image(t2_path)

        similarity = torch.sum(
            t1_features * t2_features,
            dim=-1
        ).item()

        return {
            "region_id": region_id,
            "t1_image": Path(t1_path).name,
            "t2_image": Path(t2_path).name,
            "crop_type": "csdnet_region_crop",
            "cosine_similarity": round(similarity, 6),
            "semantic_difference": round(1.0 - similarity, 6)
        }

    def compare_all_regions(self):

        start = time.perf_counter()

        metadata_path = self.region_dir / "regions.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Region metadata not found: {metadata_path}"
            )

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        results = []

        for region in metadata["regions"]:

            region_id = region["region_id"]

            t1_path = (
                self.region_dir / f"{region_id}_T1.png"
            )

            t2_path = (
                self.region_dir / f"{region_id}_T2.png"
            )

            if not t1_path.exists() or not t2_path.exists():

                print(
                    "Skipping:",
                    region_id,
                    "- image pair missing"
                )

                continue

            result = self.compare_pair(
                t1_path,
                t2_path,
                region_id
            )

            results.append(result)

            print(
                f"{region_id} | "
                f"Similarity: {result['cosine_similarity']:.4f} | "
                f"Semantic difference: "
                f"{result['semantic_difference']:.4f}"
            )

        output_file = (
            self.output_dir / "temporal_comparison.json"
        )

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        print("\nTemporal comparison completed.")
        print("Regions compared:", len(results))
        print("Results saved to:", output_file)

        print(
            "Processing time:",
            round(time.perf_counter() - start, 3),
            "seconds"
        )

        return results


if __name__ == "__main__":

    comparator = RemoteCLIPTemporalComparison()

    comparator.compare_all_regions()
