"""
model.py: Custom Neural Network Architecture for Lane Segmentation from Scratch
Author: Peeranat Chunhok (Student ID: 6710110295)
Course: 241-353 Artificial Intelligence Ecosystem Module
Department: Artificial Intelligence Engineering, Faculty of Engineering, Prince of Songkla University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block consisting of:
    [Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU]
    Provides rich feature extraction while maintaining spatial dimensions via padding=1.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, dropout_rate: float = 0.0):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block:
    [MaxPool2d(2) -> DoubleConv]
    Reduces spatial resolution by half while expanding feature channels.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block:
    [ConvTranspose2d / Bilinear Upsample -> Skip Connection Concat -> DoubleConv]
    Recovers spatial resolution and fuses high-level semantic context with low-level spatial detail.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle potential padding differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        # Skip connection concatenation along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output Projection block:
    [1x1 Conv2d] maps final feature channels to target class logits (1 channel for binary lane mask).
    """
    def __init__(self, in_channels: int, out_channels: int = 1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CustomLaneUNet(nn.Module):
    """
    CustomLaneUNet: Custom Encoder-Decoder Architecture for Lane Segmentation from Scratch.
    
    Architecture Highlights:
    - Symmetrical 4-level contracting and expanding paths
    - Direct Skip Connections preserving sub-pixel lane boundary positions
    - Batch Normalization in all convolutional stages for smooth gradient flow
    - Memory-efficient feature channel configuration (base_c=32)
    - Fully trained from scratch without any pretrained backbones
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_c: int = 32, bilinear: bool = False):
        super(CustomLaneUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, base_c)               # Level 1: 3 -> base_c (e.g. 32)
        self.down1 = Down(base_c, base_c * 2)                   # Level 2: 32 -> 64
        self.down2 = Down(base_c * 2, base_c * 4)               # Level 3: 64 -> 128
        self.down3 = Down(base_c * 4, base_c * 8)               # Level 4: 128 -> 256
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)    # Bottleneck: 256 -> 512 (or 256)
        
        # Expanding Path (Decoder with Skip Connections)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear) # Level 4 -> 3
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)  # Level 3 -> 2
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)  # Level 2 -> 1
        self.up4 = Up(base_c * 2, base_c, bilinear)                # Level 1 -> Output
        
        # Segmentation Head (1x1 Conv)
        self.outc = OutConv(base_c, num_classes)
        
        # Initialize all layers from scratch
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming (He) Normal Initialization for Conv layers and Constant initialization for BatchNorm.
        Essential for stable convergence when training entirely from scratch without pretrained weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder forward pass with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Binary segmentation logits
        logits = self.outc(x)
        return logits


def get_model(in_channels: int = 3, num_classes: int = 1, base_c: int = 32, bilinear: bool = False) -> CustomLaneUNet:
    """Helper factory function to instantiate CustomLaneUNet."""
    return CustomLaneUNet(in_channels=in_channels, num_classes=num_classes, base_c=base_c, bilinear=bilinear)


if __name__ == "__main__":
    # Test model instantiation and forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(in_channels=3, num_classes=1, base_c=32).to(device)
    
    # Test input shapes
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    dummy_out = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"CustomLaneUNet created successfully!")
    print(f"Device: {device}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_out.shape}")
    print(f"Total Parameters: {total_params:,} ({total_params * 4 / (1024**2):.2f} MB)")
    print(f"Trainable Parameters: {trainable_params:,}")
