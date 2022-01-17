from dataset.transform import crop, hflip, normalize, resize, blur, cutout, resize_crop

import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import yaml

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, split_file_path=None, pseudo_mask_path=None,reliable=None,val_split="val_split_0"):
        """
        :param name: dataset name, pascal, melanoma or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param split_file_path: path of yaml file for splits.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(split_file_path,'r') as file:
                split_dict = yaml.load(file, Loader=yaml.FullLoader)[val_split]
                self.labeled_ids = split_dict["labeled"]
                if reliable is None:          
                    self.unlabeled_ids = split_dict["unlabeled"]
                elif reliable is True:
                    self.unlabeled_ids = split_dict["reliable"]
                elif reliable is False:
                    self.unlabeled_ids = split_dict["unreliable"]
                #multiply label to match the cound of unlabled
                self.ids = \
                    self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids
        elif mode =='test':
            with open(split_file_path,'r') as file:
                self.ids = yaml.load(file, Loader=yaml.FullLoader)
        else:
            with open(split_file_path) as file:
                split_dict = yaml.load(file, Loader=yaml.FullLoader)[val_split]
                if mode == 'val':
                    self.ids = split_dict["val"]
                elif mode == 'label':
                    if reliable is None:
                        self.ids = split_dict["unlabeled"]
                    elif reliable is True:
                        self.ids = split_dict["reliable"]
                    elif reliable is False:
                        self.ids = split_dict["unreliable"]
                elif mode == 'train':
                    self.ids = split_dict["labeled"]  

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0]))

        if self.mode == 'val' or self.mode == 'label'  or self.mode == 'test':
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
            img, mask = resize_crop(img, mask, self.size)
            img, mask = normalize(img, mask)
            #print(img.cpu().numpy().shape)
            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # basic augmentation on all training images
        base_size = 256#400 if self.name == 'pascal' else 256 if self.name == 'melanoma' else 2048
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' and id in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)

        img, mask = normalize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.ids)
