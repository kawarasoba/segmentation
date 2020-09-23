from albumentations.augmentations import transforms
from albumentations import Compose

import numpy as np
import constants as cons

def HueSaturationValue(hue_shift_limit=10,sat_shift_limit=30,val_shift_limit=20,p=0.5):
    return transforms.HueSaturationValue(
        hue_shift_limit=hue_shift_limit,
        sat_shift_limit=sat_shift_limit,
        val_shift_limit=val_shift_limit,
        p=p)

def RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True,p=0.5):
    return transforms.RandomBrightnessContrast(
        brightness_limit=brightness_limit,
        contrast_limit=contrast_limit,
        brightness_by_max=brightness_by_max,
        p=p)

def CLAHE(clip_limit=8.0, tile_grid_size=(4,4), always_apply=False,p=0.5):
    return transforms.CLAHE(
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        always_apply=always_apply,
        p=p)

def JpegCompression(quality_lower=90,quality_upper=100,p=0.5):
    return transforms.JpegCompression(
        quality_lower=quality_lower,
        quality_upper=quality_upper,
        p=p)

def GaussNoise(var_limit=(10.0,50.0),mean=0,p=0.5):
    return transforms.GaussNoise(
        var_limit=var_limit,
        mean=mean,
        p=p)

def MedianBlur(blur_limit=7,p=0.5):
    return transforms.MedianBlur(
        blur_limit=blur_limit,
        p=p)

def ElasticTransform(alpha=1,sigma=100,alpha_affine=50,p=0.5): 
    return transforms.ElasticTransform(
        alpha=alpha,
        sigma=sigma,
        alpha_affine=alpha_affine,
        p=p)

def HorizontalFlip(p=0.5):
    return transforms.HorizontalFlip(p=p)

def Rotate(limit=15, interpolation=1,border_mode=4,p=0.5):
    return transforms.Rotate(
        limit=limit, 
        interpolation=interpolation,
        border_mode=border_mode,
        p=p)

def CoarseDropout(max_holes=1,max_height=30,max_width=30,fill_value=0,p=0.5):
    return transforms.CoarseDropout(
        max_holes=max_holes,
        max_height=max_height,
        max_width=max_width,
        fill_value=fill_value,
        p=p)

def RandomSizedCrop(min_max_height=(150,224),height=cons.IMAGE_SIZE,width=cons.IMAGE_SIZE,p=0.5):
    return transforms.RandomSizedCrop(
        min_max_height=min_max_height,
        height=height,
        width=width,
        p=p)

def Resize(height=cons.IMAGE_SIZE,width=cons.IMAGE_SIZE):
    return transforms.Resize(
        height=height,
        width=width
        )

def SmallestMaxSize(max_size=cons.IMAGE_SIZE):
    return transforms.SmallestMaxSize(
        max_size=224
        )

def CenterCrop(height=cons.IMAGE_SIZE,width=cons.IMAGE_SIZE):
    return transforms.CenterCrop(
        height=cons.IMAGE_SIZE,
        width=cons.IMAGE_SIZE
        )

def Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),max_pixel_value=255.0):
    return transforms.Normalize(
        mean=mean,
        std=std,
        max_pixel_value=max_pixel_value
        )

transform_train=Compose([
    Resize(height=cons.IMAGE_SIZE,width=cons.IMAGE_SIZE),
    CLAHE(clip_limit=8.0, tile_grid_size=(4,4), always_apply=False,p=0.5),
    GaussNoise(var_limit=(10.0,50.0),mean=0,p=0.5),
    Rotate(limit=15, interpolation=1,border_mode=4,p=0.5),
    Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),max_pixel_value=255.0)
    ])

transform_valid=Compose([
    Resize(height=cons.IMAGE_SIZE,width=cons.IMAGE_SIZE),
    Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),max_pixel_value=255.0)
    ],p=1.0)

transform_inference=Compose([
    SmallestMaxSize(max_size=224),
    CenterCrop(height=cons.IMAGE_SIZE,width=cons.IMAGE_SIZE),
    Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),max_pixel_value=255.0)
    ],p=1.0)

def main():
    print(transform_train)
    print(transform_valid)

if __name__ == '__main__':
    main()

