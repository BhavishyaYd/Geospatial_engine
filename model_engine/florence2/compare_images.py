from pathlib import Path
from PIL import Image

from florence2.model import load_florence2
from florence2.caption import generate_caption


def main():

    project_root = Path(__file__).resolve().parent.parent

    image_t1_path = (
        project_root / "data" / "LEVIR-CD" /
        "test" / "A" / "test_1.png"
    )

    image_t2_path = (
        project_root / "data" / "LEVIR-CD" /
        "test" / "B" / "test_1.png"
    )

    output_dir = project_root / "outputs" / "florence2"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_t1 = Image.open(image_t1_path).convert("RGB")
    image_t2 = Image.open(image_t2_path).convert("RGB")

    model, processor, device = load_florence2()

    print("\nGenerating T1 caption...")

    caption_t1 = generate_caption(
        model, processor, image_t1, "<DETAILED_CAPTION>"
    )

    print("\nGenerating T2 caption...")

    caption_t2 = generate_caption(
        model, processor, image_t2, "<DETAILED_CAPTION>"
    )

    print("\nT1 Caption:")
    print(caption_t1)

    print("\nT2 Caption:")
    print(caption_t2)

    report = (
        "FLORENCE-2 TEMPORAL IMAGE DESCRIPTION\n"
        "=====================================\n\n"
        f"T1 Caption:\n{caption_t1}\n\n"
        f"T2 Caption:\n{caption_t2}\n\n"
        "Note: Caption differences are not verified change detections.\n"
    )

    output_file = output_dir / "test_1_temporal_report.txt"

    output_file.write_text(report, encoding="utf-8")

    print("\nReport saved to:", output_file)


if __name__ == "__main__":
    main()