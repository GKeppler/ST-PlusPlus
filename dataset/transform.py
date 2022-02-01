import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms


def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=max(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=max(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, abs(w - size))
    y = random.randint(0, abs(h - size))
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, base_size, ratio_range):
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask

#center crop to sqaure, then base_size
def resize_crop(img, mask, base_size):
    w, h = img.size
    if h > w:
        crop_size = w
    else:
        crop_size = h
    left = (w - crop_size)/2
    top = (h - crop_size)/2
    right = (w + crop_size)/2
    bottom = (h + crop_size)/2
    #make it sqaure
    img = img.crop((left, top, right, bottom))
    mask = mask.crop((left, top, right, bottom))

    #resize to base_size
    img = img.resize((base_size, base_size), Image.BILINEAR)
    mask = mask.resize((base_size, base_size), Image.NEAREST)

    return img, mask

def downsample(img, mask, base_size):
    w, h = img.size

    if h > w:
        oh = base_size
        ow = int(1.0 * w * base_size / h + 0.5)
    else:
        ow = base_size
        oh = int(1.0 * h * base_size / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask


# from https://github.com/marinbenc/medical-polar-training/blob/main/polar_transformations.py
import cv2
import numpy as np
def to_polar(img, mask, center=None):
    img = np.float32(img)
    mask = np.float32(mask)
    value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
    if center is None:
        center = (img.shape[0]//2, img.shape[1]//2)
    polar_image = cv2.linearPolar(img, center, value, cv2.WARP_FILL_OUTLIERS)  
    polar_mask = cv2.linearPolar(mask, center, value, cv2.WARP_FILL_OUTLIERS) 
    polar_image = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    polar_mask = cv2.rotate(polar_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    polar_image = Image.fromarray(polar_image.astype('uint8'))
    polar_mask = Image.fromarray(polar_mask.astype('uint8'))
    return polar_image, polar_mask

def to_cart(polar_image, polar_mask, center=None):
    polar_image = np.float32(polar_image)
    polar_mask = np.float32(polar_mask)
    polar_image = cv.rotate(polar_image, cv.ROTATE_90_CLOCKWISE)
    polar_mask = cv.rotate(polar_mask, cv.ROTATE_90_CLOCKWISE)
    if center is None:
        center = (polar_image.shape[1]//2, polar_image.shape[0]//2)
    value = np.sqrt(((polar_image.shape[1]/2.0)**2.0)+((polar_image.shape[0]/2.0)**2.0))
    img = cv.linearPolar(polar_image, center, value, cv.WARP_FILL_OUTLIERS + cv.WARP_INVERSE_MAP)
    mask = cv.linearPolar(polar_mask, center, value, cv.WARP_FILL_OUTLIERS + cv.WARP_INVERSE_MAP)
    img = Image.fromarray(img.astype('uint8'))
    mask = Image.fromarray(mask.astype('uint8'))
    return img, mask
