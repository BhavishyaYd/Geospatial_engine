
import torch
from PIL import Image
from remoteclip.model import load_remoteclip

def compare_image_with_text(image_path, descriptions):

    model, preprocess, tokenizer, device = load_remoteclip()

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    text_tokens = tokenizer(descriptions).to(device)

    with torch.inference_mode():

        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

        similarities = (image_features @ text_features.T)[0]

    results = sorted(
        zip(descriptions, similarities.cpu().tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nImage–Text Similarity Results")
    print("-" * 45)

    for description, score in results:
        print(f"{score:.4f} : {description}")

    return results


if __name__ == "__main__":

    image_path = "data/images/airport.jpg"

    descriptions = [
        "An airport with aircraft on the runway",
        "A residential area with multiple buildings",
        "A dense green forest",
        "A large water body",
        "A highway with moving vehicles"
    ]

    compare_image_with_text(image_path, descriptions)
