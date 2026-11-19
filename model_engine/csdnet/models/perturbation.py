
import torch
import torch.nn as nn


class PerturbationModule(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16,
        swap_ratio=0.1,
        mask_ratio=0.1
    ):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.swap_ratio = swap_ratio
        self.mask_ratio = mask_ratio

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1)
        )

    def perturb_features(self, feature_t1, feature_t2):

        channels = feature_t1.shape[1]

        if not self.training:
            return feature_t1, feature_t2

        swap_count = int(channels * self.swap_ratio)
        mask_count = int(channels * self.mask_ratio)

        if swap_count > 0:
            indices = torch.randperm(
                channels, device=feature_t1.device
            )[:swap_count]

            temp_t1 = feature_t1.clone()
            temp_t2 = feature_t2.clone()

            temp_t1[:, indices] = feature_t2[:, indices]
            temp_t2[:, indices] = feature_t1[:, indices]

            feature_t1 = temp_t1
            feature_t2 = temp_t2

        if mask_count > 0:
            indices_t1 = torch.randperm(
                channels, device=feature_t1.device
            )[:mask_count]

            indices_t2 = torch.randperm(
                channels, device=feature_t2.device
            )[:mask_count]

            feature_t1 = feature_t1.clone()
            feature_t2 = feature_t2.clone()

            feature_t1[:, indices_t1] = 0
            feature_t2[:, indices_t2] = 0

        return feature_t1, feature_t2

    def forward(self, feature_t1, feature_t2):

        perturbed_t1, perturbed_t2 = self.perturb_features(
            feature_t1, feature_t2
        )

        difference = perturbed_t1 - perturbed_t2

        pooled_difference = torch.nn.functional.adaptive_avg_pool2d(
            difference, 1
        )

        attention = torch.sigmoid(
            self.mlp(pooled_difference) + self.mlp(difference)
        )

        enhanced_t1 = feature_t1 + attention * feature_t1
        enhanced_t2 = feature_t2 + attention * feature_t2

        return enhanced_t1, enhanced_t2, difference, attention


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    channels_list = [64, 128, 256, 512]

    for channels in channels_list:

        module = PerturbationModule(channels).to(device)
        module.train()

        feature_t1 = torch.randn(
            1, channels, 32, 32, device=device
        )

        feature_t2 = torch.randn(
            1, channels, 32, 32, device=device
        )

        with torch.inference_mode():

            output_t1, output_t2, difference, attention = module(
                feature_t1, feature_t2
            )

        print(
            f"Channels: {channels} | "
            f"T1: {tuple(output_t1.shape)} | "
            f"T2: {tuple(output_t2.shape)} | "
            f"Attention: {tuple(attention.shape)}"
        )

    print("Perturbation Module test successful.")
