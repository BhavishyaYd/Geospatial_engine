
from pathlib import Path
import torch
from PIL import Image, ImageDraw

from florence2.model import load_florence2
from florence2.caption import generate_caption


def generate_region_proposals(model, processor, image):
    device = next(model.parameters()).device

    inputs = processor(
        text="<REGION_PROPOSAL>",
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(
            dtype=next(model.parameters()).dtype
        )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    result = processor.post_process_generation(
        generated_text,
        task="<REGION_PROPOSAL>",
        image_size=(image.width, image.height)
    )

    return result["<REGION_PROPOSAL>"]["bboxes"]


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def main():
    project_root = Path(__file__).resolve().parent.parent

    test_dir = project_root / "data" / "LEVIR-CD" / "test"

    t1_path = test_dir / "A" / "test_1.png"
    t2_path = test_dir / "B" / "test_1.png"

    output_dir = (
        project_root / "outputs" / "florence2" /
        "matched_region_analysis"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    image_t1 = Image.open(t1_path).convert("RGB")
    image_t2 = Image.open(t2_path).convert("RGB")

    if image_t1.size != image_t2.size:
        raise ValueError("T1 and T2 image dimensions must match.")

    model, processor, device = load_florence2()

    # Generate proposals independently
    print("\nGenerating T1 proposals...")
    boxes_t1 = generate_region_proposals(
        model, processor, image_t1
    )

    print("T1 proposals:", len(boxes_t1))

    print("\nGenerating T2 proposals...")
    boxes_t2 = generate_region_proposals(
        model, processor, image_t2
    )

    print("T2 proposals:", len(boxes_t2))

    if not boxes_t1 or not boxes_t2:
        print("\nNo matching regions available.")
        return

    # Find the pair with maximum IoU
    best_pair = max(
        (
            (calculate_iou(box1, box2), box1, box2)
            for box1 in boxes_t1
            for box2 in boxes_t2
        ),
        key=lambda item: item[0]
    )

    best_iou, box_t1, box_t2 = best_pair

    print("\nBest matching region pair:")
    print("T1 box:", box_t1)
    print("T2 box:", box_t2)
    print("IoU:", round(best_iou, 4))

    # Create a common crop using the union of both boxes
    padding = 80

    crop_box = (
        max(0, int(min(box_t1[0], box_t2[0]) - padding)),
        max(0, int(min(box_t1[1], box_t2[1]) - padding)),
        min(image_t1.width, int(max(box_t1[2], box_t2[2]) + padding)),
        min(image_t1.height, int(max(box_t1[3], box_t2[3]) + padding))
    )

    crop_t1 = image_t1.crop(crop_box)
    crop_t2 = image_t2.crop(crop_box)

    crop_t1_path = output_dir / "matched_region_T1.png"
    crop_t2_path = output_dir / "matched_region_T2.png"

    crop_t1.save(crop_t1_path)
    crop_t2.save(crop_t2_path)

    print("\nCommon crop coordinates:", crop_box)
    print("Crop size:", crop_t1.size)

    # Generate descriptions for the same geographical area
    print("\nGenerating T1 region description...")

    description_t1 = generate_caption(
        model,
        processor,
        crop_t1,
        "<DETAILED_CAPTION>"
    )

    print("\nGenerating T2 region description...")

    description_t2 = generate_caption(
        model,
        processor,
        crop_t2,
        "<DETAILED_CAPTION>"
    )

    print("\nT1 description:")
    print(description_t1)

    print("\nT2 description:")
    print(description_t2)

    # Draw matched boxes on original images
    annotated_t1 = image_t1.copy()
    annotated_t2 = image_t2.copy()

    ImageDraw.Draw(annotated_t1).rectangle(
        box_t1, outline="red", width=4
    )

    ImageDraw.Draw(annotated_t2).rectangle(
        box_t2, outline="red", width=4
    )

    annotated_t1.save(output_dir / "matched_T1_annotated.png")
    annotated_t2.save(output_dir / "matched_T2_annotated.png")

    # Save report
    report = f"""
FLORENCE-2 MATCHED TEMPORAL REGION ANALYSIS
===========================================

T1 Image: {t1_path.name}
T2 Image: {t2_path.name}

T1 Proposals: {len(boxes_t1)}
T2 Proposals: {len(boxes_t2)}

MATCHED REGIONS
---------------

T1 Bounding Box: {box_t1}
T2 Bounding Box: {box_t2}

Intersection over Union (IoU): {best_iou:.4f}

Common Crop Coordinates: {crop_box}

T1 DESCRIPTION
--------------
{description_t1}

T2 DESCRIPTION
--------------
{description_t2}

IMPORTANT
---------
The descriptions are model-generated interpretations.
They are not verified change detections.

The IoU measures bounding-box overlap, not object identity.
CSD-Net predictions must be used for change localization.
"""

    report_path = output_dir / "matched_region_report.txt"

    report_path.write_text(report, encoding="utf-8")

    print("\nMatched temporal analysis completed!")
    print("Output directory:", output_dir)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
