from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from model.semseg.base import BaseNet
from model.semseg.unet import Unet
from model.semseg.small_unet import SmallUnet
from utils import count_params

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from dataset.isic_dermo_data_module import IsicDermoDataModule

MODE = None
global step_train
global step_val
step_train = 0
step_val = 0
use_wandb = False


def parse_args():
    parser = argparse.ArgumentParser(description="ST and ST++ Framework")

    # basic settings
    parser.add_argument(
        "--data-root",
        type=str,
        default="/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pascal", "cityscapes", "melanoma", "pneumothorax", "breastCancer"],
        default="breastCancer",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "resnet50", "resnet101"],
        default="resnet50",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["deeplabv3plus", "pspnet", "deeplabv2", "unet", "smallUnet"],
        default="smallUnet",
    )
    parser.add_argument(
        "--val-split", type=str, default="val_split_0"
    )  # need to implement in Dataloader, crrently not working

    # semi-supervised settings
    parser.add_argument("--split", type=str, default="1_30")
    parser.add_argument("--shuffle", type=int, default=0)
    # these are derived from the above split, shuffle and dataset. They dont need to be set
    parser.add_argument(
        "--split-file-path", type=str, default=None
    )  # "dataset/splits/melanoma/1_30/split_0/split_sample.yaml")
    parser.add_argument(
        "--test-file-path", type=str, default=None
    )  # "dataset/splits/melanoma/test_sample.yaml")
    parser.add_argument("--pseudo-mask-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--reliable-id-path", type=str, default=None)
    parser.add_argument(
        "--plus",
        dest="plus",
        default=False,
        action="store_true",
        help="whether to use ST++",
    )
    parser.add_argument(
        "--use-wandb", default=False, help="whether to use WandB for logging"
    )
    parser.add_argument(
        "--use-tta", default=True, help="whether to use Test Time Augmentation"
    )

    args = parser.parse_args()

    # autoparse? bzw use ******LightningCLI*********

    # add model specific args
    parser = BaseNet.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    return args


def main(args):
    if use_wandb:
        wandb.init(project="ST++", entity="gkeppler")
        wandb_logger = WandbLogger(project="ST++")
        wandb.define_metric("step_train")
        wandb.define_metric("step_val")
        wandb.define_metric("step_epoch")
        wandb.define_metric("Pictures", step_metric="step_epoch")
        wandb.define_metric("loss", step_metric="step_train")
        wandb.define_metric("mIOU", step_metric="step_val")
        wandb.config.update(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit("Please specify reliable-id-path in ST++.")

    dataModule = IsicDermoDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        train_yaml_path=args.split_file_path,
        test_yaml_path=args.test_file_path,
        pseudo_mask_path=args.pseudo_mask_path,
    )
    num_classes = {"pascal": 21, "cityscapes": 19, "melanoma": 2, "breastCancer": 3}[
        args.dataset
    ]
    model_zoo = {
        "deeplabv3plus": DeepLabV3Plus,
        "pspnet": PSPNet,
        "deeplabv2": DeepLabV2,
        "unet": Unet,
        "smallUnet": SmallUnet,
    }
    model = model_zoo[args.model](backbone=args.backbone, nclass=num_classes, args=args)

    # saves a file like: my/path/sample-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("./", f"{args.save_path}"),
        filename=f"{args.model}" + "-{epoch:02d}-{val_acc:.2f}",
        mode="max",
        save_weights_only=True,
    )

    dev_run = False  # not working when predicting with best_model checkpoint
    Trainer = pl.Trainer.from_argparse_args(
        args,
        fast_dev_run=dev_run,
        max_epochs=args.epochs,
        log_every_n_steps=2,
        logger=wandb_logger if args.use_wandb else None,
        callbacks=[checkpoint_callback],
        # gpus=[0],
        accelerator="cpu",
    )
    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print(
        "\n================> Total stage 1/%i: "
        "Supervised training on labeled images (SupOnly)" % (6 if args.plus else 3)
    )

    Trainer.fit(model=model, datamodule=dataModule)

    if not args.plus:
        print("\nParams: %.1fM" % count_params(model))

        """
            ST framework without selective re-training
        """
        # <============================= Pseudolabel all unlabeled images =============================>
        print(
            "\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images"
        )

        Trainer.predict(
            datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path
        )

        # <======================== Re-training on labeled and unlabeled images ========================>
        print(
            "\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images"
        )

        model = model_zoo[args.model](
            backbone=args.backbone, nclass=num_classes, args=args
        )
        # increase max epochs to double the amount
        Trainer.fit_loop.max_epochs *= 2
        dataModule.mode = "semi_train"
        Trainer.fit(
            model=model, datamodule=dataModule
        )
        return


if __name__ == "__main__":
    args = parse_args()

    if args.lr is None:
        args.lr = 0.001
    if args.epochs is None:
        args.epochs = {"pascal": 80, "cityscapes": 240, "melanoma": 80}[args.dataset]
    # if args.lr is None:
    #     args.lr = {'pascal': 0.001, 'cityscapes': 0.004, 'melanoma': 0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {
            "pascal": 321,
            "cityscapes": 721,
            "melanoma": 256,
            "breastCancer": 256,
        }[args.dataset]
    if args.split_file_path is None:
        args.split_file_path = f"dataset/splits/{args.dataset}/{args.split}/split_{args.shuffle}/split.yaml"
    if args.test_file_path is None:
        args.test_file_path = f"dataset/splits/{args.dataset}/test.yaml"
    if args.pseudo_mask_path is None:
        args.pseudo_mask_path = (
            f"outdir/pseudo_masks/{args.dataset}/{args.split}/split_{args.shuffle}"
        )
    if args.save_path is None:
        args.save_path = (
            f"outdir/models/{args.dataset}/{args.split}/split_{args.shuffle}"
        )
    if args.reliable_id_path is None:
        args.reliable_id_path = (
            f"outdir/reliable_ids/{args.dataset}/{args.split}/split_{args.shuffle}"
        )
    print()
    print(args)

    main(args)
