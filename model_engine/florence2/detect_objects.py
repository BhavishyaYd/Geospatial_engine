
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from florence2.model import load_florence2


def detect_objects(model, processor, image):

    device = next(model.parameters()).device

    inputs = processor(
        text="<OD>",
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
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
        task="<OD>",
        image_size=(image.width, image.height)
    )

    return result


def main():

    project_root = Path(__file__).resolve().parent.parent

    image_path = (
        project_root / "data" / "LEVIR-CD" /
        "test" / "B" / "test_1.png"
    )

    output_dir = project_root / "outputs" / "florence2"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    model, processor, device = load_florence2()

    print("\nRunning object detection...")

    results = detect_objects(model, processor, image)

    print("\nDetection results:")
    print(results)

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    detections = results["<OD>"]

    for box, label in zip(
        detections["bboxes"],
        detections["labels"]
    ):

        x1, y1, x2, y2 = box

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            label,
            fill="red"
        )

    output_image = output_dir / "test_1_object_detection.png"

    annotated_image.save(output_image)

    print("\nDetection image saved to:", output_image)
    print("Objects detected:", len(detections["labels"]))


if __name__ == "__main__":
    main()
