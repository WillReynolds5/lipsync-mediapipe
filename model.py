import torch
import torch.nn as nn
from vit_pytorch import SimpleViT


class ViT(nn.Module):

    def __init__(self, image_size=128, patch_size=32, num_classes=60, dim=128, depth=1, heads=4, mlp_dim=128):
        super().__init__()

        self.vit = SimpleViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )


    def forward(self, x):
        x = self.vit(x)
        return x


if __name__ == '__main__':
    model = ViT()
    x = torch.randn(1, 3, 128, 32)
    y = model(x)
    print(y.shape)
