
from pathlib import Path
from PIL import Image
from florence2.model import load_florence2
from florence2.caption import generate_caption


def main():
    project_root = Path(__file__).resolve().parent.parent

    image_path = (
        project_root / "data" / "LEVIR-CD" /
        "test" / "B" / "test_1.png"
    )

    output_dir = project_root / "outputs" / "florence2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Previously generated Florence-2 region proposal
    bbox = [466.432, 531.968, 537.088, 580.096]

    # Load original satellite image
    image = Image.open(image_path).convert("RGB")

    # Add surrounding context around the proposed region
    padding = 80

    x1, y1, x2, y2 = bbox

    crop_box = (
        max(0, int(x1 - padding)),
        max(0, int(y1 - padding)),
        min(image.width, int(x2 + padding)),
        min(image.height, int(y2 + padding))
    )

    cropped_image = image.crop(crop_box)

    print("Original image size:", image.size)
    print("Region bounding box:", bbox)
    print("Context crop coordinates:", crop_box)
    print("Cropped image size:", cropped_image.size)

    # Save cropped region
    crop_path = output_dir / "test_1_proposed_region_crop.png"
    cropped_image.save(crop_path)

    print("\nCropped region saved to:", crop_path)

    # Load Florence-2
    model, processor, device = load_florence2()

    # Generate detailed region description
    print("\nGenerating detailed region description...")

    description = generate_caption(
        model,
        processor,
        cropped_image,
        "<DETAILED_CAPTION>"
    )

    print("\nRegion Description:")
    print(description)

    # Save description
    output_file = output_dir / "test_1_region_description.txt"

    output_file.write_text(
        "FLORENCE-2 REGION DESCRIPTION\n"
        "=============================\n\n"
        f"Bounding Box: {bbox}\n"
        f"Crop Coordinates: {crop_box}\n\n"
        f"Description:\n{description}\n",
        encoding="utf-8"
    )

    print("\nDescription saved to:", output_file)


if __name__ == "__main__":
    main()
