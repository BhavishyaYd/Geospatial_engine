
from pathlib import Path
import torch
from PIL import Image

from florence2.model import load_florence2


def generate_caption(model, processor, image, task):

    prompt = task

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    device = next(model.parameters()).device

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
            max_new_tokens=256,
            num_beams=3
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    result = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height)
    )

    return result


def main():

    project_root = Path(__file__).resolve().parent.parent

    image_path = project_root / "data" / "LEVIR-CD" / "test" / "B" / "test_1.png"

    output_dir = project_root / "outputs" / "florence2"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading image:", image_path)

    image = Image.open(image_path).convert("RGB")

    print("Image size:", image.size)

    model, processor, device = load_florence2()

    print("\nGenerating caption...")

    caption = generate_caption(
        model,
        processor,
        image,
        "<CAPTION>"
    )

    print("\nBasic caption:")
    print(caption)

    print("\nGenerating detailed caption...")

    detailed_caption = generate_caption(
        model,
        processor,
        image,
        "<DETAILED_CAPTION>"
    )

    print("\nDetailed caption:")
    print(detailed_caption)

    output_file = output_dir / "levir_test_1_t2_caption.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Basic Caption:\n")
        file.write(str(caption))

        file.write("\n\nDetailed Caption:\n")
        file.write(str(detailed_caption))

    print("\nCaptions saved to:", output_file)


if __name__ == "__main__":
    main()

