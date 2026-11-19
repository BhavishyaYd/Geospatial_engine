
import torch
from PIL import Image

from remoteclip.model import load_remoteclip


def generate_image_embedding(image_path):

    model, preprocess, tokenizer, device = load_remoteclip()

    image = Image.open(image_path).convert("RGB")

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():

        image_features = model.encode_image(image_tensor)

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

    embedding = image_features.cpu().numpy()

    print("Image embedding generated successfully!")
    print("Embedding shape:", embedding.shape)
    print("Embedding dimension:", embedding.shape[1])
    print("First 10 values:", embedding[0][:10])

    return embedding


if __name__ == "__main__":

    image_path = "data/images/airport.jpg"

    generate_image_embedding(image_path)
