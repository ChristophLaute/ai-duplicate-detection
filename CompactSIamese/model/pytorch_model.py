import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, dropout_p=0.5, T=15):
        super().__init__()
        self.encoder = Encoder()
        self.prediction_head = nn.Linear(T, 1)
        self.dropout_p = dropout_p

    def forward(self, x1, x2):
        B, T, C, H, W = x1.shape

        z1 = self.encoder(x1.reshape(B * T, C, H, W)).reshape(B, T, 64)
        z2 = self.encoder(x2.reshape(B * T, C, H, W)).reshape(B, T, 64)

        if self.training and self.dropout_p > 0:
            mask = (torch.rand_like(z1) > self.dropout_p).float() / (1 - self.dropout_p)
            z1 = z1 * mask
            z2 = z2 * mask

        distances = torch.sqrt(torch.sum((z1 - z2) ** 2, dim=2) + 1e-8)  # (B, T)
        logits = self.prediction_head(distances).squeeze(-1)               # (B,)

        return logits, z1, z2

    def encode(self, x):
        B, T, C, H, W = x.shape
        z = self.encoder(x.reshape(B * T, C, H, W)).reshape(B, T, 64)
        return z
