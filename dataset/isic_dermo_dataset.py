import glob
import os
import cv2
import numpy as np
from dataset.transform import crop, hflip, normalize, resize, blur, cutout
import random
from PIL import Image
from torchvision import transforms
from dataset.partially_labeled_dataset import (
    PartiallyLabeledDataset,
)

class IsicDermoDataset(PartiallyLabeledDataset):
    """
    :param root_dir: root path of the dataset.
    :param id_path: path of labeled or  unlabeled image ids
    :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
    """
    def __init__(
        self,
        root_dir: str,
        id_path=None,
        pseudo_mask_path="todo",
        samples_data_format="jpg",
        labels_data_format="png",
        transforms=None,
        empty_dataset=False,
        insert_bg_class=False,
        labels_available=True,
        return_trafos=False,
        
    ):
        self.root_dir = root_dir
        self.id_path = id_path
        self.samples_data_format = samples_data_format
        self.labels_data_format = labels_data_format
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        self.samples_dir = os.path.join(self.root_dir, 'images')
        self.insert_bg_class = insert_bg_class
        self.labels_available = labels_available
        self.return_trafos = return_trafos
        self.transforms = transforms
        if transforms is None:
                self.transforms = lambda x, y: (x,y,0)
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [self.transforms]

        self.indices = []
        if not empty_dataset:
            if self.id_path is not None:
                with open(id_path, 'r') as f:
                    self.indices = f.read().splitlines()
        self.raw_mode = False

    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        #if file with image ids is available, use this file
        if self.id_path is None:        
            img_path = os.path.join(self.samples_dir,
                    f"ISIC_{self.indices[idx]}.{self.samples_data_format}")
        else:
            id = self.indices[idx]
            img_path = os.path.join(self.root_dir, id.split(' ')[0])
            
        img = Image.open(img_path)

        label = None
        if self.labels_available:
            if self.id_path is None:
                label_path = os.path.join(
                    self.labels_dir, 
                    f"ISIC_{self.indices[idx]}_segmentation.{self.labels_data_format}"
                )
            else:
                label_path = os.path.join(self.root_dir, id.split(' ')[1])
            label = Image.open(label_path)
        else:
            #open image as mask for label for shorter code
            label = Image.open(img_path)
                
        #apply standart transformations
        base_size = 256#400 if self.name == 'pascal' else 256 if self.name == 'melanoma' else 2048
        img, label = resize(img, label, base_size, (0.5, 2.0))
        img, label = crop(img, label, 100)
        img, label = hflip(img, label, p=0.5)
        img, label = normalize(img, label)    
        
        # strong augmentation on unlabeled images
        if self.labels_available is False:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img = cutout(img, label, p=0.5)   
        
        if self.labels_available:
            return img, label 
        return img

        # raw mode -> no transforms
        if self.raw_mode:
            if self.labels_available:
                return sample_img,label
            else:
                return sample_img
        
        sample_img_lst = []
        label_lst = []
        trafo_lst = []
        for transform in self.transforms:
            im, lbl, trafo = transform(sample_img, label)
            sample_img_lst.append(im)
            label_lst.append(lbl)
            trafo_lst.append(trafo)

        if len(sample_img_lst) == 1:
            sample_img_lst = sample_img_lst[0]
            label_lst = label_lst[0] if len(label_lst) > 0 else label_lst
            trafo_lst = trafo_lst[0] if len(trafo_lst) > 0 else trafo_lst
        
        # sample_img_lst (optional: labels) (optional: trafos)
        if not self.return_trafos and not self.labels_available:
            return sample_img_lst
        if self.return_trafos and not self.labels_available:
            return sample_img_lst, trafo_lst
        if not self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst
        if self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst, trafo_lst
    
    # def get_samples(self):
    #     return self.indices

    # def pop_sample(self, index):
    #     return self.indices.pop(index)

    # def add_sample(self, new_sample):
    #     return self.indices.append(new_sample)
