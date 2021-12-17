from abc import ABC, ABCMeta, abstractmethod
import logging
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from ALS4L.deep_learning_components.data_processing.augmentations.semantic_segmentation.custom_collate_fcn import (
#     custom_collate,
# )
# from ALS4L.deep_learning_components.data_processing.augmentations.base.dataloader_worker_init import seed_worker

class PartiallyLabeledLightningDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            num_workers,
            pin_memory,
            shuffle,
            drop_last,
        ):
        super().__init__()
        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        logging.info(f"Using {self.num_workers} workers for data loading")
    

    @abstractmethod 
    def train_dataloader(self, get_labeled_share=True):
        if get_labeled_share:
            return self._labeled_train_dataloader()
        return self._unlabeled_train_dataloader()

    @abstractmethod 
    def _labeled_train_dataloader(self):
        print(
            f"Getting Labeled Train Dataset with \
                length {len(self.labeled_train_dataset)}"
        )
        return DataLoader(
            self.labeled_train_dataset,
            batch_size=self.batch_size,
            #collate_fn=custom_collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            #worker_init_fn=seed_worker,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )

    @abstractmethod 
    def _unlabeled_train_dataloader(self):
        print(
            f"Getting Unlabeled Train Dataset with \
                length {len(self.unlabeled_train_dataset)}"
        )
        return DataLoader(
            self.unlabeled_train_dataset,
            batch_size=self.batch_size,
            #collate_fn=custom_collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            #worker_init_fn=seed_worker,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )

    @abstractmethod 
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            #worker_init_fn=seed_worker
        )

    @abstractmethod   
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            #worker_init_fn=seed_worker
        )

 