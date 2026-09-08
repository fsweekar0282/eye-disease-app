import timm
import torch.nn as nn


class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0,
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(1536),
            nn.Dropout(0.4),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))
