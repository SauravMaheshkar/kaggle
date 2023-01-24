"""Image Augmentation Utility Functions"""
from typing import Dict, List

import albumentations as A
import cv2


def get_data_transforms(image_size: List[int]) -> Dict:
    """
    Get Augmentations for training and validation

    :param image_size: Image Size for resizing augmentation
    :type image_size: List[int]
    :return: A Dictionary with various augmentations for training
            and validation
    :rtype: Dict
    """
    data_transforms: Dict = {
        "train": A.Compose(
            [
                A.Resize(*image_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.05, rotate_limit=60, p=0.5
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
            ],
            p=1.0,
        ),
        "valid": A.Compose(
            [
                A.Resize(*image_size, interpolation=cv2.INTER_NEAREST),
            ],
            p=1.0,
        ),
    }

    return data_transforms
