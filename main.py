import torch
from icecream import ic
from torch import nn

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

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    test = torch.tensor([1, 2, 3])
    ic(test)
