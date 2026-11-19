
import torch
from remoteclip.model import load_remoteclip

def generate_text_embedding(text):
    model, preprocess, tokenizer, device = load_remoteclip()

    text_tokens = tokenizer([text]).to(device)

    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

    embedding = text_features.cpu().numpy()

    print("Text embedding generated successfully!")
    print("Embedding shape:", embedding.shape)
    print("Embedding dimension:", embedding.shape[1])
    print("First 10 values:", embedding[0][:10])

    return embedding


if __name__ == "__main__":
    query = "An airport with aircraft on the runway"
    generate_text_embedding(query)
