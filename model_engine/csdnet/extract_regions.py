
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw


def extract_regions(
    image_t1_path,
    image_t2_path,
    mask_path,
    output_dir,
    min_area=30,
    padding=32
):

    image_t1 = Image.open(image_t1_path).convert("RGB")
    image_t2 = Image.open(image_t2_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L")

    if image_t1.size != image_t2.size:
        raise ValueError("T1 and T2 dimensions must match.")

    if image_t1.size != mask_image.size:
        raise ValueError("Image and mask dimensions must match.")

    image_width, image_height = image_t1.size

    mask = np.array(mask_image)
    binary_mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    draw = ImageDraw.Draw(image_t2)

    regions = []
    region_number = 0

    for component_id in range(1, num_labels):

        x, y, width, height, area = stats[component_id]

        if area < min_area:
            continue

        region_number += 1

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_width, x + width + padding)
        y2 = min(image_height, y + height + padding)

        crop_t1 = image_t1.crop((x1, y1, x2, y2))
        crop_t2 = image_t2.crop((x1, y1, x2, y2))

        component_mask = (
            labels[y:y + height, x:x + width] == component_id
        ).astype(np.uint8) * 255

        crop_mask = Image.fromarray(component_mask)

        region_name = f"region_{region_number:02d}"

        crop_t1.save(output_dir / f"{region_name}_T1.png")
        crop_t2.save(output_dir / f"{region_name}_T2.png")
        crop_mask.save(output_dir / f"{region_name}_mask.png")

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="yellow",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 18)),
            region_name,
            fill="yellow"
        )

        regions.append({
            "region_id": region_name,
            "component_id": int(component_id),
            "bbox": [
                int(x), int(y),
                int(x + width), int(y + height)
            ],
            "crop_bbox": [
                int(x1), int(y1),
                int(x2), int(y2)
            ],
            "area_pixels": int(area),
            "centroid": [
                round(float(centroids[component_id][0]), 2),
                round(float(centroids[component_id][1]), 2)
            ]
        })

        print(
            f"{region_name}: "
            f"area={area}, "
            f"bbox=({x}, {y}, {width}, {height})"
        )

    image_t2.save(output_dir / "regions_annotated.png")

    report = {
        "image_dimensions": [image_width, image_height],
        "minimum_component_area": min_area,
        "padding": padding,
        "total_components": num_labels - 1,
        "accepted_regions": len(regions),
        "regions": regions
    }

    with open(
        output_dir / "regions.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nRegion extraction completed.")
    print("Total detected components:", num_labels - 1)
    print("Accepted regions:", len(regions))
    print("Output directory:", output_dir)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image-t1", required=True)
    parser.add_argument("--image-t2", required=True)
    parser.add_argument("--mask", required=True)

    parser.add_argument(
        "--output",
        default="outputs/csdnet_inference/regions"
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=30
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=32
    )

    args = parser.parse_args()

    extract_regions(
        args.image_t1,
        args.image_t2,
        args.mask,
        args.output,
        args.min_area,
        args.padding
    )


if __name__ == "__main__":
    main()
