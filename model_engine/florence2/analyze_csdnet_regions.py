
import json
import argparse

from pathlib import Path
from PIL import Image

from florence2.model import load_florence2
from florence2.caption import generate_caption


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REGION_DIR = (
    PROJECT_ROOT / "outputs" / "csdnet_inference" / "regions"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "florence2" / "csdnet_context_analysis"
)

CONTEXT_SIZE = 256


def extract_context(image, bbox, context_size=CONTEXT_SIZE):

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    width, height = image.size

    crop_width = min(context_size, width)
    crop_height = min(context_size, height)

    left = center_x - crop_width // 2
    top = center_y - crop_height // 2

    left = max(0, min(left, width - crop_width))
    top = max(0, min(top, height - crop_height))

    right = left + crop_width
    bottom = top + crop_height

    return image.crop((left, top, right, bottom)), [
        left, top, right, bottom
    ]


def get_caption(result):

    if isinstance(result, dict):
        return result.get("<DETAILED_CAPTION>", str(result))

    return str(result)


def analyze_regions(
    image_t1_path,
    image_t2_path,
    region_dir=DEFAULT_REGION_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    context_size=CONTEXT_SIZE
):

    region_dir = Path(region_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = region_dir / "regions.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Region metadata not found: {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        region_data = json.load(file)

    regions = region_data["regions"]

    if not regions:
        print("No candidate regions found.")

        return {
            "model": "Florence-2-base",
            "total_regions_analyzed": 0,
            "results": []
        }

    image_t1 = Image.open(image_t1_path).convert("RGB")
    image_t2 = Image.open(image_t2_path).convert("RGB")

    if image_t1.size != image_t2.size:
        raise ValueError(
            "T1 and T2 dimensions must match."
        )

    print("Candidate regions:", len(regions))
    print("Context crop size:", context_size)

    model, processor, device = load_florence2()

    results = []

    try:

        for index, region in enumerate(regions, start=1):

            region_id = region["region_id"]

            print("\n" + "=" * 55)
            print(
                f"Analyzing {region_id} "
                f"({index}/{len(regions)})"
            )
            print("=" * 55)

            crop_t1, crop_bbox = extract_context(
                image_t1,
                region["bbox"],
                context_size
            )

            crop_t2, _ = extract_context(
                image_t2,
                region["bbox"],
                context_size
            )

            crop_t1.save(
                output_dir / f"{region_id}_T1_context.png"
            )

            crop_t2.save(
                output_dir / f"{region_id}_T2_context.png"
            )

            print("Generating T1 description...")

            caption_t1 = get_caption(
                generate_caption(
                    model,
                    processor,
                    crop_t1,
                    "<DETAILED_CAPTION>"
                )
            )

            print("Generating T2 description...")

            caption_t2 = get_caption(
                generate_caption(
                    model,
                    processor,
                    crop_t2,
                    "<DETAILED_CAPTION>"
                )
            )

            result = {
                "region_id": region_id,
                "original_bbox": region["bbox"],
                "context_bbox": crop_bbox,
                "area_pixels": region["area_pixels"],
                "t1_description": caption_t1,
                "t2_description": caption_t2
            }

            results.append(result)

            print("\nT1:", caption_t1)
            print("\nT2:", caption_t2)

    finally:

        del model
        del processor

        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    report = {
        "model": "Florence-2-base",
        "region_source": "CSD-Net predicted change mask",
        "context_size": context_size,
        "description_reliability": "Supplementary; requires visual verification",
        "total_regions_analyzed": len(results),
        "results": results
    }

    json_path = output_dir / "context_analysis.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    text_path = output_dir / "context_analysis.txt"

    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("CSD-Net + Florence-2 Context Analysis\n")
        file.write("=" * 55 + "\n\n")

        for result in results:

            file.write(
                f"Region: {result['region_id']}\n"
            )

            file.write(
                f"Original bbox: {result['original_bbox']}\n"
            )

            file.write(
                f"Context bbox: {result['context_bbox']}\n"
            )

            file.write("\nT1 Description:\n")
            file.write(result["t1_description"] + "\n")

            file.write("\nT2 Description:\n")
            file.write(result["t2_description"] + "\n")

            file.write("-" * 55 + "\n\n")

    print("\nContext analysis completed!")
    print("Regions analyzed:", len(results))
    print("JSON:", json_path)
    print("Text:", text_path)

    return report


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-t1",
        required=True
    )

    parser.add_argument(
        "--image-t2",
        required=True
    )

    parser.add_argument(
        "--regions",
        default=str(DEFAULT_REGION_DIR)
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=CONTEXT_SIZE
    )

    args = parser.parse_args()

    analyze_regions(
        image_t1_path=args.image_t1,
        image_t2_path=args.image_t2,
        region_dir=args.regions,
        output_dir=args.output,
        context_size=args.context_size
    )


if __name__ == "__main__":
    main()
