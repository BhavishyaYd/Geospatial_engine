
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
    / "remoteclip_semantic"
)

UNCERTAINTY_MARGIN = 0.02


PROMPTS = {

    "residential": [
        "An aerial satellite image of residential buildings and houses.",
        "An overhead view of a populated residential neighborhood.",
        "A remote sensing image showing houses and urban settlements."
    ],

    "roads": [
        "An aerial satellite image of roads, streets and highways.",
        "An overhead view of road networks and transportation routes.",
        "A remote sensing image showing paved roads and intersections."
    ],

    "vegetation": [
        "An aerial satellite image of trees, forests and green vegetation.",
        "An overhead view of green areas covered with plants and trees.",
        "A remote sensing image showing natural vegetation and forest cover."
    ],

    "bare_land": [
        "An aerial satellite image of empty land, soil and open ground.",
        "An overhead view of exposed soil and undeveloped land.",
        "A remote sensing image showing barren land without buildings."
    ],

    "industrial": [
        "An aerial satellite image of industrial buildings and factories.",
        "An overhead view of warehouses and industrial facilities.",
        "A remote sensing image showing large industrial structures."
    ],

    "water": [
        "An aerial satellite image of rivers, lakes and water bodies.",
        "An overhead view of a water surface and surrounding land.",
        "A remote sensing image showing rivers, ponds and lakes."
    ]
}


class RemoteCLIPSemanticAnalysis:

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

        self.categories = list(PROMPTS.keys())

        self.prepare_text_features()

        print("\nRemoteCLIP semantic analysis initialized.")
        print("Device:", self.device)
        print("Categories:", len(self.categories))
        print("Prompts per category:", 3)

        print(
            "Initialization time:",
            round(time.perf_counter() - start, 3),
            "seconds"
        )

    def prepare_text_features(self):

        self.text_features = []

        with torch.inference_mode():

            for category in self.categories:

                descriptions = PROMPTS[category]

                tokens = self.tokenizer(descriptions).to(
                    self.device
                )

                features = self.model.encode_text(tokens)

                features = features / features.norm(
                    dim=-1,
                    keepdim=True
                )

                category_embedding = features.mean(
                    dim=0,
                    keepdim=True
                )

                category_embedding = category_embedding / (
                    category_embedding.norm(
                        dim=-1,
                        keepdim=True
                    )
                )

                self.text_features.append(category_embedding)

            self.text_features = torch.cat(
                self.text_features,
                dim=0
            )

    def encode_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.preprocess(image).unsqueeze(0).to(
            self.device
        )

        with torch.inference_mode():

            image_features = self.model.encode_image(image_tensor)

            image_features = image_features / (
                image_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

            similarities = (
                image_features @ self.text_features.T
            )[0]

        return similarities.cpu().tolist()

    def analyze_image(self, image_path):

        scores = self.encode_image(image_path)

        results = []

        for category, score in zip(
            self.categories,
            scores
        ):

            results.append({
                "category": category,
                "similarity": round(float(score), 6)
            })

        results.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        top_category = results[0]["category"]

        top_score = results[0]["similarity"]

        second_score = results[1]["similarity"]

        margin = round(
            top_score - second_score,
            6
        )

        if margin < UNCERTAINTY_MARGIN:

            predicted_category = "uncertain"
            classification_status = "low_margin"

        else:

            predicted_category = top_category
            classification_status = "provisional"

        return {
            "image": Path(image_path).name,
            "predicted_category": predicted_category,
            "highest_scoring_category": top_category,
            "top_score": top_score,
            "second_score": second_score,
            "score_margin": margin,
            "classification_status": classification_status,
            "category_scores": results
        }

    def analyze_all_regions(self):

        start = time.perf_counter()

        metadata_path = self.region_dir / "regions.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Region metadata not found: {metadata_path}"
            )

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        all_results = []

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

            t1_result = self.analyze_image(t1_path)

            t2_result = self.analyze_image(t2_path)

            both_provisional = (
                t1_result["classification_status"] == "provisional"
                and
                t2_result["classification_status"] == "provisional"
            )

            if both_provisional:

                category_changed = (
                    t1_result["predicted_category"]
                    != t2_result["predicted_category"]
                )

                transition = (
                    f"{t1_result['predicted_category']}"
                    f"_to_"
                    f"{t2_result['predicted_category']}"
                )

            else:

                category_changed = None
                transition = "uncertain"

            result = {
                "region_id": region_id,
                "T1": t1_result,
                "T2": t2_result,
                "category_changed": category_changed,
                "semantic_transition": transition
            }

            all_results.append(result)

            print("\n", region_id)

            print(
                "T1:",
                t1_result["predicted_category"],
                "| Margin:",
                t1_result["score_margin"]
            )

            print(
                "T2:",
                t2_result["predicted_category"],
                "| Margin:",
                t2_result["score_margin"]
            )

            print("Transition:", transition)

        output_file = (
            self.output_dir / "semantic_analysis_multprompt.json"
        )

        with open(output_file, "w", encoding="utf-8") as file:

            json.dump(
                all_results,
                file,
                indent=4
            )

        print("\nMulti-prompt semantic analysis completed.")

        print("Regions analyzed:", len(all_results))

        print("Results saved to:", output_file)

        print(
            "Processing time:",
            round(time.perf_counter() - start, 3),
            "seconds"
        )

        return all_results


if __name__ == "__main__":

    analyzer = RemoteCLIPSemanticAnalysis()

    analyzer.analyze_all_regions()
