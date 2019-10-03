#!/usr/bin/env python3

import numpy as np
import random
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image


def largest_subset(img, dims):
    # First, crop the image so it has the right aspect ratio
    w, h = img.size  # PIL uses (w,h)
    if (w / dims[0]) > (h / dims[1]):  # If the image is too short and wide
        cropped_dims = (h, h * dims[0] / dims[1])  # Pytorch uses (h,w)
    else:
        cropped_dims = (w * dims[1] / dims[0], w)
    img = transforms.CenterCrop(cropped_dims)(img)

    # Next, rescale the image
    img = transforms.Resize(dims)(img)

    return img


def load_images(top_dir, dims):
    pairs = ImageFolder(
        root=top_dir,
        # transform=transforms.Compose(
        # [
        # transforms.CenterCrop((200, 200)),
        # transforms.Resize((100, 100)),
        # transforms.ToTensor(),
        # ]
        # ),
    )

    transformed = []
    for img, cat in pairs:
        img = transforms.ToTensor()(largest_subset(img, dims))
        transformed.append((img, cat))

    random.shuffle(transformed)  # Get rid of any order
    return transformed


if __name__ == "__main__":
    target_dims = (120, 128)  # height, width
    pairs = load_images("dataset1", target_dims)
