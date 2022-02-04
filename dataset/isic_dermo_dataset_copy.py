import os
from dataset.transform import crop, hflip, normalize, resize, blur, cutout
import random
from PIL import Image
import math

class IsicDermoDataset():
    """
    :param root_dir: root path of the dataset.
    :param id_path: path of labeled or  unlabeled image ids
    :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
    """  
    def __init__(
        self,
        root_dir: str,
        mode: str,
        base_size = 256,
        crop_size = 256,
        labeled_id_list=None,
        unlabeled_id_list=None,
        pseudo_mask_path=None,  
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.base_size = base_size
        self.size = crop_size
        self.unlabeled_id_list = unlabeled_id_list
        self.labeled_id_list = labeled_id_list
        self.pseudo_mask_path = pseudo_mask_path
   
        if mode == 'semi_train':
            self.ids = \
                self.labeled_id_list * math.ceil(len(self.unlabeled_id_list) / len(self.labeled_id_list)) + self.unlabeled_id_list
        elif mode == 'val' or mode =='train':
            self.ids = labeled_id_list
        elif mode == 'label':
            self.ids = unlabeled_id_list

    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        img_path = os.path.join(self.root_dir, id.split(' ')[0])     
        img = Image.open(img_path)

        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.root_dir, id.split(' ')[1]))

            img, mask = resize(img, mask, self.base_size,(1,1))# (0.5, 2.0))
            img, mask = normalize(img, mask)
            #print(img.cpu().numpy().shape)
            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_id_list):
            mask = Image.open(os.path.join(self.root_dir, id.split(' ')[1]))
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # basic augmentation on all training images
        img, mask = resize(img, mask, self.base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' and id in self.unlabeled_id_list:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)

        img, mask = normalize(img, mask)

        return img, mask, 