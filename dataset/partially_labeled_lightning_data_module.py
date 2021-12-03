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
            initial_labeled_ratio,
            num_workers,
            pin_memory,
            shuffle,
            drop_last,
            initial_labeled_size=None
        ):
        super().__init__()
        self.labeled_ratio = initial_labeled_ratio
        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.initial_labeled_size = initial_labeled_size
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

    # @abstractmethod   
    # def get_unlabeled_samples(self):
    #     return self.unlabeled_train_dataset.get_samples()

    # @abstractmethod   
    # def label_sample(self, labeled_sample, unlabeled_sample_index):
    #     self.unlabeled_train_dataset.pop_sample(unlabeled_sample_index)
    #     self.labeled_train_dataset.add_sample(labeled_sample)

    # @abstractmethod   
    # def init_val_dataset(self, split_lst=None):
    #     # default init is random
    #     if split_lst is None:
    #         num_val_samples = int(round(len(self.labeled_train_dataset) * (self.val_to_train_ratio)))
    #         num_val_samples = num_val_samples if num_val_samples > 0 else 1
    #         for _ in range(
    #             num_val_samples
    #         ):
    #             self.val_dataset.add_sample(
    #                 self.labeled_train_dataset.pop_sample(
    #                     random.randrange(len(self.labeled_train_dataset))
    #                 )
    #             )

    #     else:
    #         ind_lst = []
    #         for elem in split_lst:
    #             ind_lst.append(self.labeled_train_dataset.indices[elem])

    #         for ind_elem in ind_lst:
    #             rem_ind = self.labeled_train_dataset.indices.index(ind_elem)
    #             self.val_dataset.add_sample(
    #                 self.labeled_train_dataset.pop_sample(
    #                     rem_ind
    #                 )
    #             )

    # @abstractmethod   
    # def reset_val_dataset(self):
    #     for _ in range(
    #             len(self.val_dataset)
    #         ):
    #         self.labeled_train_dataset.add_sample(
    #             self.val_dataset.pop_sample(0)
    #         )

    # @abstractmethod   
    # def assign_labeled_unlabeled_split(self):
    #     if self.initial_labeled_size is None:
    #         for _ in range(int(len(self.labeled_train_dataset) * (1 - self.labeled_ratio))):
    #             self.unlabeled_train_dataset.add_sample(
    #                 self.labeled_train_dataset.pop_sample(
    #                     random.randrange(len(self.labeled_train_dataset))
    #                 )
    #             )
    #     else:
    #         if len(self.labeled_train_dataset)>=self.initial_labeled_size:
    #             num_pops = len(self.labeled_train_dataset) - self.initial_labeled_size
    #             for _ in range(num_pops):
    #                 self.unlabeled_train_dataset.add_sample(
    #                     self.labeled_train_dataset.pop_sample(
    #                         random.randrange(len(self.labeled_train_dataset))
    #                     )
    #                 )
    #         else:
    #             raise ValueError(f"Dataset is smaller than {self.initial_labeled_size}")

    #     self.unlabeled_train_dataset.resort_samples()
 