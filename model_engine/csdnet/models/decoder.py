
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class WeightedFusion(nn.Module):

    def __init__(self, inputs, channels):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(inputs))
        self.epsilon = 1e-4
        self.conv = ConvBlock(channels)

    def forward(self, features):

        weights = F.relu(self.weights)

        weights = weights / (
            weights.sum() + self.epsilon
        )

        output = sum(
            weights[i] * features[i]
            for i in range(len(features))
        )

        return self.conv(output)


class BiFPN(nn.Module):

    def __init__(
        self,
        in_channels=(128, 256, 512, 512),
        out_channels=128
    ):
        super().__init__()

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
            for channels in in_channels
        ])

        self.top_down = nn.ModuleList([
            WeightedFusion(2, out_channels)
            for _ in range(3)
        ])

        self.bottom_up = nn.ModuleList([
            WeightedFusion(3, out_channels)
            for _ in range(3)
        ])

    def resize(self, x, target):

        return F.interpolate(
            x,
            size=target.shape[-2:],
            mode="nearest"
        )
    
    def prepare_features(self, differences, attentions, content_difference):

        if len(differences) != 4 or len(attentions) != 4:
            raise ValueError("Expected four difference and attention maps.")

        combined = []

        for i in range(4):

            difference = differences[i]
            attention = attentions[i]

            if i == 3:
                difference = difference + content_difference

            enhanced = difference * attention

            combined.append(enhanced)

        return combined

    def forward(self, features):

        if len(features) != 4:
            raise ValueError(
                "BiFPN requires exactly four feature maps."
            )

        p1, p2, p3, p4 = [
            projection(feature)
            for projection, feature in zip(
                self.projections, features
            )
        ]

        # Top-down pathway

        td4 = p4

        td3 = self.top_down[0]([
            p3,
            self.resize(td4, p3)
        ])

        td2 = self.top_down[1]([
            p2,
            self.resize(td3, p2)
        ])

        td1 = self.top_down[2]([
            p1,
            self.resize(td2, p1)
        ])

        # Bottom-up pathway

        bu1 = td1

        bu2 = self.bottom_up[0]([
            p2,
            td2,
            self.resize(bu1, p2)
        ])

        bu3 = self.bottom_up[1]([
            p3,
            td3,
            self.resize(bu2, p3)
        ])

        bu4 = self.bottom_up[2]([
            p4,
            td4,
            self.resize(bu3, p4)
        ])

        return [bu1, bu2, bu3, bu4]


class ChangePredictionHead(nn.Module):

    def __init__(self, channels=128):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 4,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.refinement = nn.Sequential(
            nn.Conv2d(
                channels,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, features, output_size):

        target_size = features[0].shape[-2:]

        resized_features = [
            F.interpolate(
                feature,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )
            for feature in features
        ]

        fused = torch.cat(resized_features, dim=1)

        fused = self.fusion(fused)

        refined = self.refinement(fused)

        logits = self.classifier(refined)

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        return logits



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = [
        torch.randn(1, 128, 64, 64, device=device),
        torch.randn(1, 128, 32, 32, device=device),
        torch.randn(1, 128, 16, 16, device=device),
        torch.randn(1, 128, 8, 8, device=device)
    ]

    model = ChangePredictionHead(
        channels=128
    ).to(device)

    model.eval()

    with torch.inference_mode():
        logits = model(
            features,
            output_size=(256, 256)
        )

    print("Device:", device)
    print("Prediction shape:", tuple(logits.shape))
    print("Prediction minimum:", logits.min().item())
    print("Prediction maximum:", logits.max().item())

    print("Change prediction head test successful.")
