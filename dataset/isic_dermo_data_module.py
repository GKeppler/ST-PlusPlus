import os
import random
import yaml
from dataset.partially_labeled_lightning_data_module import (
    PartiallyLabeledLightningDataModule,
)
from dataset.isic_dermo_dataset import IsicDermoDataset

class IsicDermoDataModule(PartiallyLabeledLightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        train_yaml_path: str,
        test_yaml_path:str,
        dataset_size = 1.0,
        shuffle=True,
        drop_last=True,
        train_transforms=None,
        train_transforms_unlabeled=None,
        val_transforms=None,
        test_transforms=None,
        return_unlabeled_trafos=False,
        num_workers=16,
        pin_memory=False,
    ):
        super().__init__(
            num_workers,
            pin_memory,
            shuffle,
            drop_last,
        )
        self.root_dir = root_dir
        self.dataset_size = dataset_size
        self.train_root_dir = os.path.join(self.root_dir, "train")
        self.test_root_dir = os.path.join(self.root_dir, "test")
        self.batch_size = batch_size
        self.train_yaml_path = train_yaml_path
        self.test_yaml_path = test_yaml_path
        self.train_transforms = train_transforms
        self.train_transforms_unlabeled = (
            train_transforms_unlabeled
            if train_transforms_unlabeled is not None
            else train_transforms
        )
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.return_unlabeled_trafos = return_unlabeled_trafos
        self.labeled_train_dataset: IsicDermoDataset = None
        self.unlabeled_train_dataset: IsicDermoDataset = None
        self.val_dataset: IsicDermoDataset = None
        self.test_dataset: IsicDermoDataset = None
        self.__init_datasets()

    def __init_datasets(self):
        with open(self.train_yaml_path,'r') as file:
            split_dict = yaml.load(file, Loader=yaml.FullLoader)
        val_split_0 = split_dict["val_split_0"]
        
        self.labeled_train_dataset = IsicDermoDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms,
            id_list = val_split_0["labeled"],
        )

        self.unlabeled_train_dataset = IsicDermoDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms_unlabeled,
            labels_available=False,
            return_trafos=self.return_unlabeled_trafos,
            id_list = val_split_0["unlabeled"]
        )

        self.val_dataset = IsicDermoDataset(
            root_dir=self.train_root_dir,
            transforms=self.val_transforms,
            id_list = val_split_0["val"]
            #empty_dataset=True,
        )

        with open(self.test_yaml_path,'r') as file:
            test_dict = yaml.load(file, Loader=yaml.FullLoader)
       
        self.test_dataset = IsicDermoDataset(
            root_dir=self.test_root_dir,
            transforms=self.test_transforms,
            id_list=test_dict
        )