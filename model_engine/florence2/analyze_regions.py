
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

    results = processor.post_process_generation(
        generated_text,
        task="<REGION_PROPOSAL>",
        image_size=(image.width, image.height)
    )

    return results["<REGION_PROPOSAL>"]


def crop_region(image, bbox, padding=80):
    x1, y1, x2, y2 = bbox

    crop_box = (
        max(0, int(x1 - padding)),
        max(0, int(y1 - padding)),
        min(image.width, int(x2 + padding)),
        min(image.height, int(y2 + padding))
    )

    cropped_image = image.crop(crop_box)

    return cropped_image, crop_box


def main():
    project_root = Path(__file__).resolve().parent.parent

    image_path = (
        project_root / "data" / "LEVIR-CD" /
        "test" / "B" / "test_1.png"
    )

    output_dir = (
        project_root / "outputs" / "florence2" /
        "dynamic_region_analysis"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading satellite image:", image_path)

    image = Image.open(image_path).convert("RGB")

    print("Image size:", image.size)

    model, processor, device = load_florence2()

    # Step 1: Generate region proposals
    print("\nGenerating region proposals...")

    proposals = generate_region_proposals(
        model, processor, image
    )

    bboxes = proposals["bboxes"]

    print("Regions proposed:", len(bboxes))

    if not bboxes:
        print("No regions were proposed.")
        return

    # Step 2: Prepare annotated image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    report = [
        "FLORENCE-2 DYNAMIC REGION ANALYSIS",
        "==================================",
        "",
        f"Image: {image_path.name}",
        f"Image dimensions: {image.size}",
        f"Total proposed regions: {len(bboxes)}",
        ""
    ]

    # Step 3: Describe each proposed region
    for index, bbox in enumerate(bboxes, start=1):

        print(f"\nProcessing region {index}/{len(bboxes)}...")

        cropped_image, crop_box = crop_region(
            image, bbox, padding=80
        )

        crop_path = (
            output_dir / f"region_{index:02d}.png"
        )

        cropped_image.save(crop_path)

        # Generate detailed caption
        description = generate_caption(
            model,
            processor,
            cropped_image,
            "<DETAILED_CAPTION>"
        )

        print("Bounding box:", bbox)
        print("Description:", description)

        # Draw bounding box on original image
        x1, y1, x2, y2 = bbox

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=4
        )

        draw.text(
            (x1, max(0, y1 - 20)),
            f"Region {index}",
            fill="red"
        )

        report.extend([
            f"REGION {index}",
            "----------",
            f"Bounding box: {bbox}",
            f"Context crop coordinates: {crop_box}",
            f"Description: {description}",
            f"Crop image: {crop_path.name}",
            ""
        ])

    # Step 4: Save annotated image
    annotated_path = output_dir / "annotated_regions.png"
    annotated_image.save(annotated_path)

    # Step 5: Save analysis report
    report_path = output_dir / "region_analysis_report.txt"

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\nRegion analysis completed!")
    print("Annotated image:", annotated_path)
    print("Analysis report:", report_path)
    print("Total regions analyzed:", len(bboxes))


if __name__ == "__main__":
    main()
