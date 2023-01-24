"""Utlity Functions for Image Handling"""
import cv2
import numpy as np


def load_img(path: str) -> np.ndarray:
    """
    Loads an Image given the path

    :param path: Path to the Image
    :type path: str
    :return: Image as a numpy array
    :rtype: np.ndarray
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    img = img.astype("float32")  # original is uint16
    maximum_value = np.max(img)
    if maximum_value:
        img /= maximum_value  # scale image to [0, 1]
    return img


def load_msk(path: str) -> np.ndarray:
    """
    Loads a mask given the path

    :param path: Path to the mask
    :type path: str
    :return: Mask as a numpy array
    :rtype: np.ndarray
    """
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    return msk
