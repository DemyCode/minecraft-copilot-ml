import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, n_unique_minecraft_blocks: int) -> None:
        super(UNet3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(n_unique_minecraft_blocks, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upconv = nn.Upsample(scale_factor=2, mode="trilinear")
        self.upconv4 = nn.Sequential(
            nn.Conv3d(1024 + 512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.upconv3 = nn.Sequential(
            nn.Conv3d(512 + 256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.upconv2 = nn.Sequential(
            nn.Conv3d(256 + 128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.upconv1 = nn.Sequential(
            nn.Conv3d(128 + 64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(64, n_unique_minecraft_blocks, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.conv1_out = self.conv1(x)
        self.conv2_out = self.conv2(self.max_pool(self.conv1_out))
        self.conv3_out = self.conv3(self.max_pool(self.conv2_out))
        self.conv4_out = self.conv4(self.max_pool(self.conv3_out))
        self.conv5_out = self.conv5(self.max_pool(self.conv4_out))
        self.upconv4_out = self.upconv4(
            torch.cat(
                (
                    self.upconv(self.conv5_out),
                    self.conv4_out,
                ),
                dim=1,
            )
        )
        self.upconv3_out = self.upconv3(
            torch.cat(
                (
                    self.upconv(self.upconv4_out),
                    self.conv3_out,
                ),
                dim=1,
            )
        )
        self.upconv2_out = self.upconv2(
            torch.cat(
                (
                    self.upconv(self.upconv3_out),
                    self.conv2_out,
                ),
                dim=1,
            )
        )
        self.upconv1_out = self.upconv1(
            torch.cat(
                (
                    self.upconv(self.upconv2_out),
                    self.conv1_out,
                ),
                dim=1,
            )
        )
        self.conv_out: torch.Tensor = self.conv(self.upconv1_out)
        return self.conv_out
