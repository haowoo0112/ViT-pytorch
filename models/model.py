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
        
    def forward(self, img):
        x = self.preprocess(img)

    def preprocess(self, img):
        y = torch.tensor([])
        for i in range(0, self.img_size, self.patch_size):
            for j in range(0, self.img_size, self.patch_size):
                x = img[:,i : i+self.patch_size, j : j+self.patch_size]
                x = x.flatten()
                y = torch.cat((y, x), 0)
                # flatten_img.append(x)
        PPC = self.channel_size*self.patch_size*self.patch_size
        y = y.view(-1, PPC)
        print("embeddimg_shape: ", y.shape)
        return y