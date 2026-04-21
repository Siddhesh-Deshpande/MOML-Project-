from __future__ import annotations

import torch
from torch import nn


class DynamicCNN(nn.Module):
    """A compact CNN whose depth and width are controlled by decision variables."""

    def __init__(
        self,
        input_resolution: int,
        n_conv_layers: int,
        conv_channels: list[int],
        n_fc_layers: int,
        hidden_units: list[int],
        dropout: float,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        assert n_conv_layers == len(conv_channels), "conv_channels length must match n_conv_layers"
        assert n_fc_layers == len(hidden_units), "hidden_units length must match n_fc_layers"

        conv_blocks: list[nn.Module] = []
        in_channels = 1

        for out_channels in conv_channels:
            conv_blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*conv_blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_resolution, input_resolution)
            feat = self.feature_extractor(dummy)
            flattened_dim = int(feat.view(1, -1).shape[1])

        classifier_layers: list[nn.Module] = []
        in_features = flattened_dim

        for units in hidden_units:
            classifier_layers.extend(
                [
                    nn.Linear(in_features, units),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_features = units

        classifier_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        flattened = features.view(features.size(0), -1)
        logits = self.classifier(flattened)
        return logits
