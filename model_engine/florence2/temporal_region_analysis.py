
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

    return result["<REGION_PROPOSAL>"]


def analyze_image(model, processor, image_path, date_name, output_dir):
    print(f"\nAnalyzing {date_name}...")

    image = Image.open(image_path).convert("RGB")

    print("Image size:", image.size)

    proposals = generate_region_proposals(
        model, processor, image
    )

    bboxes = proposals["bboxes"]

    print("Regions proposed:", len(bboxes))

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    report = [
        f"{date_name} REGION ANALYSIS",
        "=" * 35,
        f"Image: {image_path.name}",
        f"Image dimensions: {image.size}",
        f"Regions proposed: {len(bboxes)}",
        ""
    ]

    for index, bbox in enumerate(bboxes, start=1):
        print(f"Processing {date_name} region {index}...")

        x1, y1, x2, y2 = bbox
        padding = 80

        crop_box = (
            max(0, int(x1 - padding)),
            max(0, int(y1 - padding)),
            min(image.width, int(x2 + padding)),
            min(image.height, int(y2 + padding))
        )

        cropped_image = image.crop(crop_box)

        crop_path = (
            output_dir / f"{date_name.lower()}_region_{index:02d}.png"
        )

        cropped_image.save(crop_path)

        description = generate_caption(
            model,
            processor,
            cropped_image,
            "<DETAILED_CAPTION>"
        )

        print("Bounding box:", bbox)
        print("Description:", description)

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
            f"Region {index}",
            "-" * 20,
            f"Bounding box: {bbox}",
            f"Crop coordinates: {crop_box}",
            f"Description: {description}",
            f"Crop file: {crop_path.name}",
            ""
        ])

    annotated_path = (
        output_dir / f"{date_name.lower()}_annotated.png"
    )

    annotated_image.save(annotated_path)

    report.append(f"Annotated image: {annotated_path.name}")

    return {
        "date": date_name,
        "bboxes": bboxes,
        "report": report,
        "annotated_image": annotated_path.name
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    test_dir = project_root / "data" / "LEVIR-CD" / "test"

    t1_path = test_dir / "A" / "test_1.png"
    t2_path = test_dir / "B" / "test_1.png"

    output_dir = (
        project_root / "outputs" / "florence2" /
        "temporal_region_analysis"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device = load_florence2()

    t1_results = analyze_image(
        model, processor, t1_path, "T1", output_dir
    )

    t2_results = analyze_image(
        model, processor, t2_path, "T2", output_dir
    )

    combined_report = [
        "FLORENCE-2 TEMPORAL REGION ANALYSIS",
        "=" * 45,
        "",
        "T1 AND T2 REGION PROPOSALS ARE GENERATED",
        "INDEPENDENTLY. THEIR BOXES ARE NOT ASSUMED",
        "TO REPRESENT THE SAME PHYSICAL LOCATION.",
        "",
        f"T1 regions: {len(t1_results['bboxes'])}",
        f"T2 regions: {len(t2_results['bboxes'])}",
        "",
        *t1_results["report"],
        "",
        *t2_results["report"],
        "",
        "INTERPRETATION",
        "--------------",
        "Caption differences are semantic clues only.",
        "They do not independently confirm real-world changes.",
        "CSD-Net change masks must be used for change localization."
    ]

    report_path = output_dir / "temporal_region_report.txt"

    report_path.write_text(
        "\n".join(combined_report),
        encoding="utf-8"
    )

    print("\nTemporal region analysis completed!")
    print("T1 regions:", len(t1_results["bboxes"]))
    print("T2 regions:", len(t2_results["bboxes"]))
    print("Output directory:", output_dir)
    print("Combined report:", report_path)


if __name__ == "__main__":
    main()
