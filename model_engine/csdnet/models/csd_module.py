
import torch
import torch.nn as nn


class ContentStyleDecoupling(nn.Module):

    def __init__(
        self,
        channels=512,
        num_tokens=8,
        num_heads=8,
        num_layers=2
    ):
        super().__init__()

        self.channels = channels
        self.num_tokens = num_tokens

        self.pseudo_change_tokens = nn.Parameter(
            torch.randn(1, num_tokens, channels) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.reconstructor = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature_t1, feature_t2):

        batch, channels, height, width = feature_t1.shape

        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {channels}"
            )

        def decompose(feature):

            original = feature

            image_tokens = feature.flatten(2).transpose(1, 2)

            pseudo_tokens = self.pseudo_change_tokens.expand(
                batch, -1, -1
            )

            style_tokens = self.decoder(
                tgt=pseudo_tokens,
                memory=image_tokens
            )

            style_vector = style_tokens.mean(dim=1)

            style_map = style_vector.unsqueeze(-1).unsqueeze(-1)

            style_map = style_map.expand(
                -1, -1, height, width
            )

            coarse_feature = original - style_map

            coarse_tokens = coarse_feature.flatten(2).transpose(1, 2)

            content_tokens = self.decoder(
                tgt=coarse_tokens,
                memory=pseudo_tokens
            )

            content = content_tokens.transpose(
                1, 2
            ).reshape(
                batch, channels, height, width
            )

            reconstructed = self.reconstructor(
                torch.cat([content, style_map], dim=1)
            )

            return content, style_map, reconstructed

        content_t1, style_t1, reconstructed_t1 = decompose(
            feature_t1
        )

        content_t2, style_t2, reconstructed_t2 = decompose(
            feature_t2
        )

        return {
            "content_t1": content_t1,
            "content_t2": content_t2,
            "style_t1": style_t1,
            "style_t2": style_t2,
            "reconstructed_t1": reconstructed_t1,
            "reconstructed_t2": reconstructed_t2
        }


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ContentStyleDecoupling(
        channels=512,
        num_tokens=8,
        num_heads=8,
        num_layers=2
    ).to(device)

    model.eval()

    feature_t1 = torch.randn(
        1, 512, 32, 32, device=device
    )

    feature_t2 = torch.randn(
        1, 512, 32, 32, device=device
    )

    with torch.inference_mode():

        outputs = model(feature_t1, feature_t2)

    print("Device:", device)

    for name, tensor in outputs.items():
        print(f"{name}: {tuple(tensor.shape)}")

    print(
        "Pseudo-change tokens:",
        tuple(model.pseudo_change_tokens.shape)
    )

    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )

    print("CSD-Net CSD module test successful.")
