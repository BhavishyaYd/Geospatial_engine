
from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image

from csdnet.models.csdnet_base import CSDNetBase


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "csdnet" / "best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "csdnet_inference"

PATCH_SIZE = 256

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading CSD-Net...")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = CSDNetBase().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("CSD-Net loaded successfully!")

    return model, device


def preprocess_image(image):

    image = image.convert("RGB")

    array = np.array(image).astype(np.float32) / 255.0

    array = (array - MEAN) / STD

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1).contiguous()

    return tensor


def pad_image(array):

    height, width = array.shape[:2]

    pad_height = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
    pad_width = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE

    if pad_height == 0 and pad_width == 0:
        return array, height, width

    mode = "reflect" if height > 1 and width > 1 else "edge"

    padded = np.pad(
        array,
        (
            (0, pad_height),
            (0, pad_width),
            (0, 0)
        ),
        mode=mode
    )

    return padded, height, width


def predict_change(model, device, image_t1, image_t2):

    if image_t1.size != image_t2.size:
        raise ValueError(
            "T1 and T2 images must have identical dimensions."
        )

    image_t1 = image_t1.convert("RGB")
    image_t2 = image_t2.convert("RGB")

    array_t1 = np.array(image_t1)
    array_t2 = np.array(image_t2)

    padded_t1, original_height, original_width = pad_image(array_t1)
    padded_t2, _, _ = pad_image(array_t2)

    height, width = padded_t1.shape[:2]

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    total_patches = (
        (height // PATCH_SIZE) * (width // PATCH_SIZE)
    )

    patch_number = 0

    use_amp = device.type == "cuda"

    print("Original image size:", original_width, "x", original_height)
    print("Padded image size:", width, "x", height)
    print("Total patches:", total_patches)

    with torch.no_grad():

        for top in range(0, height, PATCH_SIZE):

            for left in range(0, width, PATCH_SIZE):

                patch_t1 = Image.fromarray(
                    padded_t1[
                        top:top + PATCH_SIZE,
                        left:left + PATCH_SIZE
                    ]
                )

                patch_t2 = Image.fromarray(
                    padded_t2[
                        top:top + PATCH_SIZE,
                        left:left + PATCH_SIZE
                    ]
                )

                tensor_t1 = preprocess_image(patch_t1)
                tensor_t2 = preprocess_image(patch_t2)

                tensor_t1 = tensor_t1.unsqueeze(0).to(device)
                tensor_t2 = tensor_t2.unsqueeze(0).to(device)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp
                ):

                    outputs = model(tensor_t1, tensor_t2)

                probabilities = (
                    outputs["prediction_probabilities"]
                    [0, 0]
                    .float()
                    .cpu()
                    .numpy()
                )

                probability_map[
                    top:top + PATCH_SIZE,
                    left:left + PATCH_SIZE
                ] = probabilities

                patch_number += 1

                print(
                    f"Processed patch {patch_number}/{total_patches}"
                )

    probability_map = probability_map[
        :original_height,
        :original_width
    ]

    prediction_mask = (
        probability_map > 0.5
    ).astype(np.uint8)

    return probability_map, prediction_mask


def save_results(image_t2, probability_map, prediction_mask, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    probability_image = Image.fromarray(
        (probability_map * 255).clip(0, 255).astype(np.uint8)
    )

    mask_image = Image.fromarray(
        prediction_mask * 255
    )

    probability_image.save(
        output_dir / "change_probability.png"
    )

    mask_image.save(
        output_dir / "change_mask.png"
    )

    original = np.array(image_t2.convert("RGB")).copy()

    overlay = original.copy()

    changed_pixels = prediction_mask == 1

    red = np.array([255, 0, 0], dtype=np.float32)

    overlay[changed_pixels] = (
        0.55 * original[changed_pixels].astype(np.float32)
        + 0.45 * red
    ).clip(0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(
        output_dir / "change_overlay.png"
    )

    print("\nResults saved:")
    print(output_dir / "change_probability.png")
    print(output_dir / "change_mask.png")
    print(output_dir / "change_overlay.png")


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
        "--output",
        default=str(OUTPUT_DIR / "prediction")
    )

    args = parser.parse_args()

    image_t1 = Image.open(args.image_t1).convert("RGB")
    image_t2 = Image.open(args.image_t2).convert("RGB")

    model, device = load_model()

    probability_map, prediction_mask = predict_change(
        model,
        device,
        image_t1,
        image_t2
    )

    save_results(
        image_t2,
        probability_map,
        prediction_mask,
        Path(args.output)
    )

    changed_pixels = int(prediction_mask.sum())
    total_pixels = prediction_mask.size

    print("\nCSD-Net Inference Results")
    print("-------------------------")
    print("Image dimensions:", image_t1.size)
    print("Changed pixels:", changed_pixels)
    print(
        "Changed area percentage:",
        round(changed_pixels / total_pixels * 100, 2),
        "%"
    )

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
