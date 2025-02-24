import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.isic_dermo_dataset import IsicDermoDataset
import logging


class IsicDermoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        train_yaml_path: str,
        test_yaml_path: str,
        pseudo_mask_path: str,
        num_workers=16,
        pin_memory=False,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        logging.info(f"Using {self.num_workers} workers for data loading")
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_yaml_path = train_yaml_path
        self.test_yaml_path = test_yaml_path
        self.pseudo_mask_path = pseudo_mask_path
        # transformations not used currently
        # self.train_transforms = train_transforms
        # self.train_transforms_unlabeled = (
        #     train_transforms_unlabeled
        #     if train_transforms_unlabeled is not None
        #     else train_transforms
        # )
        # self.val_transforms = val_transforms
        # self.test_transforms = test_transforms

        self.sup_train_dataset: IsicDermoDataset = None
        self.semi_train_dataset: IsicDermoDataset = None
        self.val_dataset: IsicDermoDataset = None
        self.predict_dataset: IsicDermoDataset = None
        self.test_dataset: IsicDermoDataset = None
        self.mode = "train"
        self.__init_datasets()

    def __init_datasets(self):
        with open(self.train_yaml_path, "r") as file:
            split_dict = yaml.load(file, Loader=yaml.FullLoader)
        val_split_0 = split_dict["val_split_0"]

        self.sup_train_dataset = IsicDermoDataset(
            root_dir=self.root_dir, labeled_id_list=val_split_0["labeled"], mode="train"
        )

        self.semi_train_dataset = IsicDermoDataset(
            root_dir=self.root_dir,
            labeled_id_list=val_split_0["labeled"],
            unlabeled_id_list=val_split_0["unlabeled"],
            pseudo_mask_path=self.pseudo_mask_path,
            mode="semi_train",
        )

        self.val_dataset = IsicDermoDataset(
            root_dir=self.root_dir, labeled_id_list=val_split_0["val"], mode="val"
        )

        self.predict_dataset = IsicDermoDataset(
            root_dir=self.root_dir,
            unlabeled_id_list=val_split_0["unlabeled"],
            mode="label",
        )

        with open(self.test_yaml_path, "r") as file:
            test_dict = yaml.load(file, Loader=yaml.FullLoader)

        self.test_dataset = IsicDermoDataset(
            root_dir=self.root_dir,
            # transforms=self.test_transforms,
            labeled_id_list=test_dict,
            mode="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.sup_train_dataset
            if self.mode == "train"
            else self.semi_train_dataset
            if self.mode == "semi_train"
            else None,
            batch_size=self.batch_size,
            # collate_fn=custom_collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # worker_init_fn=seed_worker,
            # shuffle=self.shuffle,
            # drop_last=self.drop_last
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            # shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # worker_init_fn=seed_worker
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            # shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # worker_init_fn=seed_worker
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            # shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # worker_init_fn=seed_worker
        )
