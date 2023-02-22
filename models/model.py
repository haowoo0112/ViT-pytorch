import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size):
        super().__init__()

        self.embeddings = Embeddings(img_size, patch_size, 3)

    def forward(self, img):
        x = self.embeddings(img)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, patch_size, channel_size):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.channel_size = channel_size
        n_patches = (img_size // patch_size) * (img_size // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 768))
        self.patch_embeddings = Conv2d(in_channels=3,
                                       out_channels=768,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(n_patches+1, 768))

    def forward(self, img):
        x = self.patch_embeddings(img)
        x = x.flatten(1)
        x = x.transpose(-1, -2)
        x = torch.cat((self.cls_token , x), dim=0)
        embeddings = x + self.position_embeddings
        print("embedding_shape: ", embeddings.shape)
