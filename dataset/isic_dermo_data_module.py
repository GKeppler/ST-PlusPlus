import os
import random
from dataset.partially_labeled_lightning_data_module import (
    PartiallyLabeledLightningDataModule,
)
from dataset.isic_dermo_dataset import IsicDermoDataset

class IsicDermoDataModule(PartiallyLabeledLightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        #val_to_train_ratio,
        labeled_id_path=None,
        unlabeled_id_path=None,
        dataset_size = 1.0,
        shuffle=True,
        drop_last=True,
        initial_labeled_ratio=None,
        train_transforms=None,
        train_transforms_unlabeled=None,
        val_transforms=None,
        test_transforms=None,
        return_unlabeled_trafos=False,
        num_workers=0,
        pin_memory=False,
        initial_labeled_size=None,
    ):
        super().__init__(
            initial_labeled_ratio,
            num_workers,
            pin_memory,
            shuffle,
            drop_last,
            initial_labeled_size
        )
        #self.val_to_train_ratio = val_to_train_ratio
        self.root_dir = root_dir
        self.labeled_id_path = labeled_id_path
        self.unlabeled_id_path = unlabeled_id_path
        self.dataset_size = dataset_size
        self.train_root_dir = os.path.join(self.root_dir, "train")
        self.test_root_dir = os.path.join(self.root_dir, "test")
        self.batch_size = batch_size
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
        self.labeled_train_dataset = IsicDermoDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms,
            id_path = self.labeled_id_path,
        )

        self.unlabeled_train_dataset = IsicDermoDataset(
            root_dir=self.train_root_dir,
            transforms=self.train_transforms_unlabeled,
            labels_available=False,
            return_trafos=self.return_unlabeled_trafos,
            id_path = self.unlabeled_id_path,
        )
        # for _ in range(int(len(self.labeled_train_dataset) * (1 - self.dataset_size))):
        #     self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))
        # for _ in range(int(len(self.labeled_train_dataset) * (1 - self.labeled_ratio))):
        #     self.unlabeled_train_dataset.add_sample(
        #         self.labeled_train_dataset.pop_sample(
        #             random.randrange(len(self.labeled_train_dataset))
        #         )
        #     )

        self.test_dataset = IsicDermoDataset(
            root_dir=self.test_root_dir,
            transforms=self.test_transforms,
        )
        
        self.val_dataset = IsicDermoDataset(
            root_dir=self.train_root_dir,
            transforms=self.val_transforms,
            #empty_dataset=True,
        )