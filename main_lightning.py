# from torch.utils.data.sampler import SubsetRandomSampler
from statistics import mode
from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from model.semseg.base import BaseNet
from model.semseg.unet import Unet
from model.semseg.small_unet import SmallUnet
from utils import count_params, meanIOU, color_map

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, base
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import wandb

# from pytorch_lightning.utilities.cli import LightningCLI
from dataset.isic_dermo_data_module import (
    IsicDermoDataModule,
)

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
    if use_wandb == True:
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

    # currently not used
    head_lr_multiple = 10.0
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
        )  # train_dataloaders = dataModule.train_dataloader(), val_dataloaders = dataModule.val_dataloader())
        return

        # best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)

    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudolabel all unlabeled images =============================>
        print(
            "\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images"
        )

        dataset = SemiDataset(
            args.dataset, args.data_root, "label", None, None, args.unlabeled_id_path
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            drop_last=False,
        )

        label(best_model, dataloader, args)

        # <======================== Re-training on labeled and unlabeled images ========================>
        print(
            "\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images"
        )

        MODE = "semi_train"

        trainset = SemiDataset(
            args.dataset,
            args.data_root,
            MODE,
            args.crop_size,
            args.labeled_id_path,
            args.unlabeled_id_path,
            args.pseudo_mask_path,
        )
        trainloader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
            drop_last=True,
        )

        model, optimizer = init_basic_elems(args)

        train(model, trainloader, valloader, criterion, optimizer, args)

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print(
        "\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training"
    )

    dataset = SemiDataset(
        args.dataset, args.data_root, "label", None, None, args.unlabeled_id_path
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print("\n\n\n================> Total stage 3/6: Pseudo labeling reliable images")

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, "reliable_ids.txt")
    dataset = SemiDataset(
        args.dataset, args.data_root, "label", None, None, cur_unlabeled_id_path
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print(
        "\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images"
    )

    MODE = "semi_train"

    trainset = SemiDataset(
        args.dataset,
        args.data_root,
        MODE,
        args.crop_size,
        args.labeled_id_path,
        cur_unlabeled_id_path,
        args.pseudo_mask_path,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        drop_last=True,
    )

    model, optimizer = init_basic_elems(args)

    best_model = train(model, trainloader, valloader, criterion, optimizer, args)

    # <=============================== Pseudo label unreliable images ================================>
    print("\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images")

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, "unreliable_ids.txt")
    dataset = SemiDataset(
        args.dataset, args.data_root, "label", None, None, cur_unlabeled_id_path
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    label(best_model, dataloader, args)

    # <================================== The 2nd stage re-training ==================================>
    print(
        "\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images"
    )

    trainset = SemiDataset(
        args.dataset,
        args.data_root,
        MODE,
        args.crop_size,
        args.labeled_id_path,
        args.unlabeled_id_path,
        args.pseudo_mask_path,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        drop_last=True,
    )

    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, criterion, optimizer, args)

    wandb.finish()


def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs
    global step_train
    global step_val

    previous_best = 0.0

    global MODE

    if MODE == "train":
        checkpoints = []

    for epoch in range(args.epochs):
        print(
            "\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f"
            % (epoch, optimizer.param_groups[0]["lr"], previous_best)
        )

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = (
                lr * 1.0 if args.model == "deeplabv2" else lr * 10.0
            )

            # wandb log with custom step
            wandb.log({"loss": loss, "step_train": step_train, "epoch": epoch})
            step_train += 1
            tbar.set_description("Loss: %.3f" % (total_loss / (i + 1)))

        metric = meanIOU(
            num_classes=21
            if args.dataset == "pascal"
            else 19
            if args.dataset == "melanoma"
            else 19
        )

        model.eval()
        tbar = tqdm(valloader)
        # set i for sample images
        i = 0
        wandb_iamges = []
        with torch.no_grad():
            for img, mask, _ in tbar:
                i = i + 1
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)
                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                # print(np.unique(np.squeeze(pred.cpu().numpy(), axis=0)))
                mIOU = metric.evaluate()[-1]
                wandb.log({"mIOU": mIOU, "step_val": step_val})

                # just some logging for wandb
                if i <= 10:
                    class_lables = dict((el, "something") for el in list(range(21)))
                    class_lables.update({255: "boarder"})
                    class_lables.update({0: "nothing"})
                    wandb_iamge = wandb.Image(
                        img,
                        masks={
                            "predictions": {
                                "mask_data": np.squeeze(pred.cpu().numpy(), axis=0),
                                "class_labels": class_lables,
                            },
                            "ground_truth": {
                                "mask_data": np.squeeze(mask.numpy(), axis=0),
                                "class_labels": class_lables,
                            },
                        },
                    )
                    wandb_iamges.append(wandb_iamge)
                tbar.set_description("mean mIOU: %.2f" % (mIOU * 100.0))
                step_val += 1

        wandb.log({"Pictures": wandb_iamges, "step_epoch": epoch})
        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(
                    os.path.join(
                        args.save_path,
                        "%s_%s_%.2f.pth" % (args.model, args.backbone, previous_best),
                    )
                )
            previous_best = mIOU
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    args.save_path, "%s_%s_%.2f.pth" % (args.model, args.backbone, mIOU)
                ),
            )

            best_model = deepcopy(model)

        if MODE == "train" and (
            (epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]
        ):
            checkpoints.append(deepcopy(model))

    if MODE == "train":
        return best_model, checkpoints

    return best_model


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(
                    args.backbone,
                    21
                    if args.dataset == "pascal"
                    else 4
                    if args.dataset == " melanoma"
                    else 19,
                )
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, "reliable_ids.txt"), "w") as f:
        for elem in id_to_reliability[: len(id_to_reliability) // 2]:
            f.write(elem[0] + "\n")
    with open(os.path.join(args.reliable_id_path, "unreliable_ids.txt"), "w") as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2 :]:
            f.write(elem[0] + "\n")


def label(model, dataloader, args):
    model.cuda()
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=21 if args.dataset == "pascal" else 19)
    cmap = color_map(args.dataset)
    i = 0
    with torch.no_grad():
        for img, mask, id in tbar:
            i += 1
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode="P")
            pred.putpalette(cmap)

            pred.save(
                "%s/%s" % (args.pseudo_mask_path, os.path.basename(id[0].split(" ")[1]))
            )

            tbar.set_description("mIOU: %.2f" % (mIOU * 100.0))
            if i > 10:
                break


def label_old(model, dataloader, args):
    model.cuda()
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=21 if args.dataset == "pascal" else 19)
    cmap = color_map(args.dataset)
    i = 0
    with torch.no_grad():
        for img, mask, id in tbar:
            i += 1
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode="P")
            pred.putpalette(cmap)

            pred.save(
                "%s/%s%s"
                % (args.pseudo_mask_path, "old", os.path.basename(id[0].split(" ")[1]))
            )

            tbar.set_description("mIOU: %.2f" % (mIOU * 100.0))
            if i > 10:
                break


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
