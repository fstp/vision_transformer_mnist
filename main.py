import random
import timeit
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from icecream import ic
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

random_seed = 42
batch_size = 512
epochs = 40
learning_rate = 1e-4
num_classes = 10
patch_size = 4
img_size = 28
in_channels = 1
num_heads = 8
dropout = 0.001
hidden_dim = 768
adam_weight_decay = 0
adam_betas = (0.9, 0.999)
activation = "gelu"
num_encoders = 4
embed_dim = (patch_size**2) * in_channels  # 16
num_patches = (img_size // patch_size) ** 2  # 49

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2),
        )
        self.cls_token = nn.Parameter(
            torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True
        )
        self.position_embeddings = nn.Parameter(
            torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_patches,
        num_classes,
        patch_size,
        embed_dim,
        num_encoders,
        num_heads,
        hidden_dim,
        dropout,
        activation,
        in_channels,
    ):
        super().__init__()
        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder_blocks = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoders
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = x[:, 0, :]
        x = self.mlp_head(x)
        return x


class TestPatchEmbedding(unittest.TestCase):
    def test_shape(self):
        model = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        ).to(device)
        x = torch.randn(512, 1, 28, 28)
        self.assertEqual(model(x).shape, (512, num_patches + 1, embed_dim))


class TestVisionTransformer(unittest.TestCase):
    def test_shape(self):
        model = VisionTransformer(
            num_patches,
            num_classes,
            patch_size,
            embed_dim,
            num_encoders,
            num_heads,
            hidden_dim,
            dropout,
            activation,
            in_channels,
        ).to(device)
        x = torch.randn(512, 1, 28, 28)
        self.assertEqual(model(x).shape, (512, num_classes))


if __name__ == "__main__":
    # model = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels).to(
    #     device
    # )
    # ic(model)
    unittest.main()
