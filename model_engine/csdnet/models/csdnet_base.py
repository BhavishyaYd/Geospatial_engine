
import torch
import torch.nn as nn

from csdnet.models.encoder import SiameseEncoder
from csdnet.models.csd_module import ContentStyleDecoupling
from csdnet.models.decoder import BiFPN, ChangePredictionHead


class CSDNetBase(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = SiameseEncoder()

        self.csd = ContentStyleDecoupling(
            channels=512,
            num_tokens=8,
            num_heads=8,
            num_layers=2
        )

        self.bifpn = BiFPN(
            in_channels=(64, 128, 256, 512),
            out_channels=128
        )

        self.prediction_head = ChangePredictionHead(
            channels=128
        )

    def forward(self, image_t1, image_t2):

        features_t1, features_t2, differences, attentions = self.encoder(
            image_t1, image_t2
        )

        csd_outputs = self.csd(
            features_t1[3],
            features_t2[3]
        )

        content_difference = torch.abs(
            csd_outputs["content_t1"] -
            csd_outputs["content_t2"]
        )

        fusion_differences = [
            torch.abs(difference)
            for difference in differences
        ]

        fusion_differences[3] = (
            fusion_differences[3] + content_difference
        )

        fused_features = self.bifpn(fusion_differences)

        prediction_logits = self.prediction_head(
            fused_features,
            output_size=image_t1.shape[-2:]
        )

        prediction_probabilities = torch.sigmoid(
            prediction_logits
        )

        prediction_mask = (
            prediction_probabilities > 0.5
        ).float()

        return {
            "prediction_logits": prediction_logits,
            "prediction_probabilities": prediction_probabilities,
            "prediction_mask": prediction_mask,
            "features_t1": features_t1,
            "features_t2": features_t2,
            "differences": differences,
            "attentions": attentions,
            "csd_outputs": csd_outputs,
            "content_difference": content_difference,
            "fused_features": fused_features
        }
