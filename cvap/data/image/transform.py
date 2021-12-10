import random
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import (
    InterpolationMode, Compose, Resize, CenterCrop, ToTensor, Normalize
)

def make_clip_image_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

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

class SharedImageTransform:
    def __init__(self, n_px):
        self.transform = transforms.Compose([
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
        ])

    def __call__(self, x):
        return self.transform(x)

class SecretImageTransform:
    def __init__(self, p_g, p_s):
        self.transform = transforms.Compose([
            GaussianBlur(p=p_g),
            Solarization(p=p_s),
            transforms.ToTensor(),
            transforms.Normalize(
                #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    def __call__(self, x):
        return self.transform(x)

class AuthenticCLIPImageTransform:
    def __init__(self, n_px):
        self.transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.transform_prime = transforms.Compose([
            transforms.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.transform_eval = self.transform_prime

    def __call__(self, x, both, train):
        x = x.convert("RGB")
        if not train:
            return self.transform_eval(x), np.array([[[1]]])
        else:
            y1 = self.transform_prime(x)
            y2 = self.transform(x) if both else np.array([[[1]]])
            return y1, y2

class CLIPImageTransform:
    def __init__(self, n_px):
        self.transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
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
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.transform_prime = transforms.Compose([
            transforms.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
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
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.transform_eval = transforms.Compose([
            transforms.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __call__(self, x, both, train):
        x = x.convert("RGB")
        if not train:
            return self.transform_eval(x), np.array([[[1]]])
        else:
            y1 = self.transform_prime(x)
            y2 = self.transform(x) if both else np.array([[[1]]])
            return y1, y2

class BarlowImageTransform:
    def __init__(self, n_px):
        self.transform = transforms.Compose([
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
        self.transform_eval = transforms.Compose([
            transforms.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __call__(self, x, both, train):
        x = x.convert("RGB")
        if not train:
            return self.transform_eval(x), np.array([[[1]]])
        else:
            y1 = self.transform_prime(x)
            y2 = self.transform(x) if both else np.array([[[1]]])
            return y1, y2
