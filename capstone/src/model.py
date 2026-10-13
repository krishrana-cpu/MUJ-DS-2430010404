# =========================================================
# model.py
# =========================================================
# Defines the custom CNN architecture, built from scratch
# (no pretrained weights). Kept in its own file so train.py
# and test.py can both import it -- this is standard practice
# for a "reproducible project structure".
# =========================================================

import torch.nn as nn


class SkinDiseaseCNN(nn.Module):
    """
    Custom CNN for skin disease classification.

    Architecture:
        4 convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool),
        each doubling the number of filters while halving spatial size,
        followed by a fully-connected classifier head with dropout.

    Input:  RGB image, IMAGE_SIZE x IMAGE_SIZE (default 128x128)
    Output: raw class scores (logits) for each disease category
    """

    def __init__(self, num_classes, image_size=128):
        super(SkinDiseaseCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # After 4 poolings, spatial size = image_size / 16
        reduced_size = image_size // 16

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * reduced_size * reduced_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x
