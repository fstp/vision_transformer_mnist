import argparse
import os
import random
import re
import timeit
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
num_heads = 16
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


class MnistTrainDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)
        return {"image": image, "label": label, "index": index}


class MnistValDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)
        return {"image": image, "label": label, "index": index}


class MnistSubmitDataset(Dataset):
    def __init__(self, images, indices):
        self.images = images
        self.indices = indices
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        index = self.indices[idx]
        image = self.transform(image)
        return {"image": image, "index": index}


class Unittests(unittest.TestCase):
    def test_patch_embeddings_shape(self):
        model = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        ).to(device)
        x = torch.randn(512, 1, 28, 28)
        self.assertEqual(model(x).shape, (512, num_patches + 1, embed_dim))

    def test_vision_trainsformer_shape(self):
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


def create_datasets():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=random_seed, shuffle=True
    )

    f, axarr = plt.subplots(1, 3)

    train_dataset = MnistTrainDataset(
        train_df.iloc[:, 1:].values.astype(np.uint8),
        train_df.iloc[:, 0].values,
        train_df.index.values,
    )
    axarr[0].imshow(train_dataset[0]["image"].squeeze(), cmap="gray")
    axarr[0].set_title("Train Image")

    val_dataset = MnistValDataset(
        val_df.iloc[:, 1:].values.astype(np.uint8),
        val_df.iloc[:, 0].values,
        val_df.index.values,
    )
    axarr[1].imshow(val_dataset[0]["image"].squeeze(), cmap="gray")
    axarr[1].set_title("Val Image")

    test_dataset = MnistSubmitDataset(
        test_df.values.astype(np.uint8), test_df.index.values
    )
    axarr[2].imshow(test_dataset[0]["image"].squeeze(), cmap="gray")
    axarr[2].set_title("Test Image")

    # plt.show()
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def create_dataloaders(datasets):
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
    )
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def create_model():
    return VisionTransformer(
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


def create_optimizer(model):
    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        weight_decay=adam_weight_decay,
    )


def save_checkpoint(epoch, model, optimizer, loss, accuracy):
    Path("checkpoints").mkdir(exist_ok=True)

    print(f"Random seed: {random_seed}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Num classes: {num_classes}")
    print(f"Patch size: {patch_size}")
    print(f"Image size: {img_size}")
    print(f"In channels: {in_channels}")
    print(f"Num heads: {num_heads}")
    print(f"Dropout: {dropout}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Adam weight decay: {adam_weight_decay}")
    print(f"Adam betas: {adam_betas}")
    print(f"Activation: {activation}")
    print(f"Num encoders: {num_encoders}")
    print(f"Embed dim: {embed_dim}")
    print(f"Num patches: {num_patches}")

    print(f"Epoch: {epoch}")
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print("Saving checkpoint...")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
        },
        f"checkpoints/checkpoint_{epoch + 1}.tar",
    )
    print("-" * 30)


def training_loop(model, optimizer, dataloader):
    train_dataloader = dataloader["train"]
    val_dataloader = dataloader["val"]

    epochs_per_checkpoint = 5
    print(f"Epochs per checkpoint: {epochs_per_checkpoint}")
    criterion = nn.CrossEntropyLoss()
    start = timeit.default_timer()

    for epoch in tqdm(range(epochs), position=0, leave=True):
        model.train()
        train_labels = []
        train_preds = []
        train_running_loss = 0
        for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_label["image"].float().to(device)
            label = img_label["label"].type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
        train_loss = train_running_loss / len(train_dataloader)

        model.eval()
        val_labels = []
        val_preds = []
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_label in enumerate(
                tqdm(val_dataloader, position=0, leave=True)
            ):
                img = img_label["image"].float().to(device)
                label = img_label["label"].type(torch.uint8).to(device)
                y_pred = model(img)
                y_pred_label = torch.argmax(y_pred, dim=1)

                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())

                loss = criterion(y_pred, label)
                val_running_loss += loss.item()
        val_loss = val_running_loss / len(val_dataloader)
        train_accuracy = sum(
            1 for x, y in zip(train_labels, train_preds) if x == y
        ) / len(train_labels)
        val_accuracy = sum(1 for x, y in zip(val_labels, val_preds) if x == y) / len(
            val_labels
        )
        print("-" * 30)
        print(f"Train loss epoch {epoch+1}: {train_loss:.4f}")
        print(f"Valid loss epoch {epoch+1}: {val_loss:.4f}")
        print(f"Train accuracy epoch {epoch+1}: {train_accuracy:.4f}")
        print(f"Valid accuracy epoch {epoch+1}: {val_accuracy:.4f}")
        print("-" * 30)
        if (epoch + 1 < epochs) and (epoch + 1) % epochs_per_checkpoint == 0:
            save_checkpoint(
                epoch,
                model,
                optimizer,
                {"train": train_loss, "val": val_loss},
                {"train": train_accuracy, "val": val_accuracy},
            )

    # Always save the final model
    save_checkpoint(
        epochs - 1,
        model,
        optimizer,
        {"train": train_loss, "val": val_loss},
        {"train": train_accuracy, "val": val_accuracy},
    )

    stop = timeit.default_timer()
    print(f"Training complete, time: {stop - start:.2f} seconds")


def find_latest_checkpoint_file(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Regular expression to match files named checkpoint_<number>
    pattern = re.compile(r"^checkpoint_(\d+).tar$")

    # Initialize variables to keep track of the latest checkpoint
    latest_number = -1
    latest_checkpoint_file = None

    # Iterate over the files and find the latest checkpoint number
    for file in files:
        match = pattern.match(file)
        if match:
            number = int(match.group(1))
            if number > latest_number:
                latest_number = number
                latest_checkpoint_file = file

    return latest_checkpoint_file


def do_test():
    checkpoint_file = find_latest_checkpoint_file("checkpoints")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_file}")
    model = create_model()
    optimizer = create_optimizer(model)
    checkpoint = torch.load(f"checkpoints/{checkpoint_file}", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["loss"]["train"]
    val_loss = checkpoint["loss"]["val"]
    train_accuracy = checkpoint["accuracy"]["train"]
    val_accuracy = checkpoint["accuracy"]["val"]
    print(f"Epoch: {epoch}")
    print(f"Train loss: {train_loss}")
    print(f"Val loss: {val_loss}")
    print(f"Train accuracy: {train_accuracy}")
    print(f"Val accuracy: {val_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        do_test()
        exit(0)

    model = create_model()
    optimizer = create_optimizer(model)
    dataset = create_datasets()
    dataloader = create_dataloaders(dataset)
    print(f"Start training (device={device})")
    training_loop(model, optimizer, dataloader)
    torch.cuda.empty_cache()
    # unittest.main()
