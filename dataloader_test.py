#from torch.utils.data.sampler import SubsetRandomSampler
from dataset.semi import SemiDataset

from copy import deepcopy
import numpy as np
import os
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
import pytorch_lightning as pl

dataset="melanoma"
data_root=r"/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/ISIC_Demo_2017"
batch_size = 2
crop_size = 100

from dataset.isic_dermo_data_module import (
    IsicDermoDataModule,
)

dataModule = IsicDermoDataModule(
        root_dir = data_root,
        batch_size = batch_size,
        train_yaml_path="dataset/splits/melanoma/1_8/split_0/split.yaml",
        test_yaml_path="dataset/splits/melanoma/test.yaml",  
)

test = dataModule.train_dataset.__getitem__(2)
test2 = dataModule.test_dataset.__getitem__(2)                                                                                                                                       
test4 = dataModule.val_dataset.__getitem__(2)

Trainer = pl.Trainer(
fast_dev_run=True,
accelerator="cpu")
from model.semseg.deeplabv3plus import DeepLabV3Plus
Trainer.fit(model=DeepLabV3Plus(backbone="resnet50", nclass=2), datamodule=dataModule)


















valset = SemiDataset(dataset, data_root, 'val', None)
valloader = DataLoader(valset, batch_size=4 if dataset == 'cityscapes' else 1,
                        shuffle=False, pin_memory=True, num_workers=6, drop_last=False)

# <====================== Supervised training with labeled images (SupOnly) ======================>
print('\n================> Total stage 1/%i: '
        'Supervised training on labeled images (SupOnly)' % (3))

trainset = SemiDataset(dataset, data_root, 'train', crop_size, labeled_id_path)
trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                            pin_memory=True, num_workers=6, drop_last=True)

