import random
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import (
    InterpolationMode, Compose, Resize, CenterCrop, ToTensor, Normalize
)

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return image.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return image

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return ImageOps.solarize(image)
        else:
            return image

class ImageTransform:
    def __init__(self, n_px):
        self.transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.RandomResizedCrop(
                n_px, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(
                #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        self.transform_prime = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.RandomResizedCrop(
                n_px, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
