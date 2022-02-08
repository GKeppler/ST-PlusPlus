from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from model.semseg.base import BaseNet
from model.semseg.unet import Unet
from utils import count_params, meanIOU, color_map, mulitmetrics

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import cv2
import yaml

MODE = None
global step_train
global step_val
step_train = 0
step_val = 0


def parse_args():
    parser = argparse.ArgumentParser(description="ST and ST++ Framework")

    # basic settings
    parser.add_argument(
        "--data-root",
        type=str,
        default="/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/ISIC_Demo_2017",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pascal", "cityscapes", "melanoma"],
        default="melanoma",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=80)
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
        choices=["deeplabv3plus", "pspnet", "deeplabv2", "unet"],
        default="unet",
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
        default=True,
        action="store_true",
        help="whether to use ST++",
    )
    parser.add_argument(
        "--use-wandb", default=True, help="whether to use WandB for logging"
    )

    args = parser.parse_args()
    return args


def main(args):
    if args.use_wandb:
        wandb.init(project="ST++", entity="gkeppler")
        wandb.run.name = (
            args.dataset
            + " "
            + args.split_file_path.split("/")[-3]
            + (" ST++" if args.plus else " ST")
        )
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

    model, optimizer = init_basic_elems(args)

    # best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)
    model.module.load_state_dict(
        torch.load(
            r"outdir/models/melanoma/1_30/split_0/unet_resnet50_87.46.pth",
            map_location="cuda:0",
        )
    )
    best_model = model
    # <====================== Test supervised model on testset (SupOnly) ======================>
    print("\n\n\n================> Test supervised model on testset (SupOnly)")
    testset = SemiDataset(
        args.dataset, args.data_root, "test", args.crop_size, args.test_file_path
    )
    testloader = DataLoader(
        testset, 1, shuffle=False, pin_memory=True, num_workers=2, drop_last=False
    )

    test(best_model, testloader, args)


def init_basic_elems(args):
    model_zoo = {
        "deeplabv3plus": DeepLabV3Plus,
        "pspnet": PSPNet,
        "deeplabv2": DeepLabV2,
        "unet": Unet,
    }
    model = model_zoo[args.model](
        args.backbone,
        21 if args.dataset == "pascal" else 2 if args.dataset == "melanoma" else 19,
    )

    head_lr_multiple = 10.0
    if args.model == "deeplabv2":
        assert args.backbone == "resnet101"
        model.load_state_dict(
            torch.load("pretrained/deeplabv2_resnet101_coco_pretrained.pth")
        )
        head_lr_multiple = 1.0

    optimizer = SGD(
        [
            {"params": model.backbone.parameters(), "lr": args.lr},
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "backbone" not in name
                ],
                "lr": args.lr * head_lr_multiple,
            },
        ],
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    model = DataParallel(model).cuda()

    return model, optimizer


def test(model, dataloader, args):
    metric = mulitmetrics(
        num_classes=21
        if args.dataset == "pascal"
        else 2
        if args.dataset == "melanoma"
        else 19
    )
    model.eval()
    tbar = tqdm(dataloader)
    # set i for sample images
    i = 0
    wandb_iamges = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for img, mask, _ in tbar:
            if args.dataset == "melanoma":
                mask = mask.clip(max=1)  # clips max value to 1: 255 to 1
            i = i + 1
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1)

            metric.add_batch(pred.cpu().numpy(), mask.numpy())
            overall_acc, mIOU, mDICE = metric.evaluate()
            tbar.set_description(
                "test mIOU: %.2f, mDICE: %.2f,overall_acc: %.2f"
                % (mIOU * 100.0, mDICE * 100.0, overall_acc * 100.0)
            )
            if args.use_wandb:
                if i <= 10:
                    # wandb.log({"img": [wandb.Image(img, caption="img")]})
                    # wandb.log({"mask": [wandb.Image(pred.cpu().numpy(), caption="mask")]})
                    class_lables = dict((el, "something") for el in list(range(21)))
                    class_lables.update({255: "boarder"})
                    class_lables.update({0: "nothing"})
                    wandb_iamge = wandb.Image(
                        cv2.resize(
                            np.moveaxis(np.squeeze(img.cpu().numpy(), axis=0), 0, -1),
                            dsize=(100, 100),
                            interpolation=cv2.INTER_NEAREST,
                        ),
                        masks={
                            "predictions": {
                                "mask_data": cv2.resize(
                                    np.squeeze(pred.cpu().numpy(), axis=0),
                                    dsize=(100, 100),
                                    interpolation=cv2.INTER_NEAREST,
                                ),
                                "class_labels": class_lables,
                            },
                            "ground_truth": {
                                "mask_data": cv2.resize(
                                    np.squeeze(mask.numpy(), axis=0),
                                    dsize=(100, 100),
                                    interpolation=cv2.INTER_NEAREST,
                                ),
                                "class_labels": class_lables,
                            },
                        },
                    )
                    wandb_iamges.append(wandb_iamge)
        if args.use_wandb:
            wandb.log({"Test Pictures": wandb_iamges})
            wandb.log(
                {
                    "test mIOU": mIOU,
                    "test mDICE": mDICE,
                    "test overall_acc": overall_acc,
                }
            )


if __name__ == "__main__":
    args = parse_args()

    if args.lr is None:
        args.lr = 0.001
    if args.epochs is None:
        args.epochs = {"pascal": 80, "cityscapes": 240, "melanoma": 80}[args.dataset]
    if args.lr is None:
        args.lr = (
            {"pascal": 0.001, "cityscapes": 0.004, "melanoma": 0.001}[args.dataset]
            / 16
            * args.batch_size
        )
    if args.crop_size is None:
        args.crop_size = {"pascal": 321, "cityscapes": 721, "melanoma": 256}[
            args.dataset
        ]

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
