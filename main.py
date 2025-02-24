from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.small_unet import SmallUnet
from model.semseg.pspnet import PSPNet
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
    parser.add_argument("--epochs", type=int, default=30)
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
        default="unet",
    )
    parser.add_argument(
        "--val-split", type=str, default="val_split_0"
    )  # need to implement in Dataloader, crrently not working

    # semi-supervised settings
    parser.add_argument("--split", type=str, default="1_4")
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
        "--use-wandb", default=False, help="whether to use WandB for logging"
    )
    parser.add_argument(
        "--use-tta", default=True, help="whether to use Test Time Augmentation"
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

    criterion = CrossEntropyLoss()  # ignore_index=255) 255 is white is melanoma
    # changed crop from None to args.crop_size
    valset = SemiDataset(
        args.dataset, args.data_root, "val", args.crop_size, args.split_file_path
    )
    valloader = DataLoader(
        valset,
        batch_size=4 if args.dataset == "cityscapes" else 1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print(
        "\n================> Total stage 1/%i: "
        "Supervised training on labeled images (SupOnly)" % (6 if args.plus else 3)
    )

    global MODE
    MODE = "train"

    trainset = SemiDataset(
        args.dataset, args.data_root, MODE, args.crop_size, args.split_file_path
    )
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        drop_last=True,
    )  # ,sampler=torch.utils.data.SubsetRandomSampler(subset_indices))

    model, optimizer = init_basic_elems(args)
    print("\nParams: %.1fM" % count_params(model))

    best_model, checkpoints = train(
        model, trainloader, valloader, criterion, optimizer, args
    )

    # <====================== Test supervised model on testset (SupOnly) ======================>
    print("\n\n\n================> Test supervised model on testset (SupOnly)")
    testset = SemiDataset(
        args.dataset, args.data_root, "test", args.crop_size, args.test_file_path
    )
    testloader = DataLoader(
        testset, 1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False
    )

    test(best_model, testloader, args)

    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print(
            "\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images"
        )

        dataset = SemiDataset(
            args.dataset, args.data_root, "label", args.crop_size, args.split_file_path
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
            args.split_file_path,
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

        # <====================== Test supervised model on testset (SupOnly) ======================>
        print("\n\n\n================> Test supervised model on testset (Re-trained)")

        test(best_model, testloader, args)

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print(
        "\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training"
    )

    dataset = SemiDataset(
        args.dataset, args.data_root, "label", args.crop_size, args.split_file_path
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

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, "reliable_ids.yaml")
    dataset = SemiDataset(
        args.dataset,
        args.data_root,
        "label",
        args.crop_size,
        cur_unlabeled_id_path,
        None,
        True,
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
        cur_unlabeled_id_path,
        args.pseudo_mask_path,
        True,
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

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, "reliable_ids.yaml")
    dataset = SemiDataset(
        args.dataset,
        args.data_root,
        "label",
        args.crop_size,
        cur_unlabeled_id_path,
        None,
        False,
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
        args.split_file_path,
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

    # <====================== Test supervised model on testset (Re-trained) ======================>
    print("\n\n\n================> Test supervised model on testset (Re-trained)")

    test(best_model, testloader, args)

    wandb.finish()


def init_basic_elems(args):
    model_zoo = {
        "deeplabv3plus": DeepLabV3Plus,
        "pspnet": PSPNet,
        "deeplabv2": DeepLabV2,
        "unet": Unet,
        "smallUnet": SmallUnet,
    }
    model = model_zoo[args.model](
        args.backbone,
        21
        if args.dataset == "pascal"
        else 2
        if args.dataset == "melanoma"
        else 2
        if args.dataset == "breastCancer"
        else 19,
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
            if args.dataset == "melanoma" or args.dataset == "breastCancer":
                mask = mask.clip(max=1)

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
            if args.use_wandb:
                wandb.log({"loss": loss, "step_train": step_train, "epoch": epoch})
            step_train += 1
            tbar.set_description("Loss: %.3f" % (total_loss / (i + 1)))

        metric = meanIOU(
            num_classes=21
            if args.dataset == "pascal"
            else 2
            if args.dataset == "melanoma"
            else 2
            if args.dataset == "breastCancer"
            else 19
        )

        model.eval()
        tbar = tqdm(valloader)
        # set i for sample images
        i = 0
        wandb_iamges = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for img, mask, _ in tbar:
                if args.dataset == "melanoma" or args.dataset == "breastCancer":
                    mask = mask.clip(max=1)
                i = i + 1
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]
                if args.use_wandb:
                    wandb.log({"mIOU": mIOU, "step_val": step_val})
                    if i <= 10:
                        # wandb.log({"img": [wandb.Image(img, caption="img")]})
                        # wandb.log({"mask": [wandb.Image(pred.cpu().numpy(), caption="mask")]})
                        class_lables = dict((el, "something") for el in list(range(21)))
                        class_lables.update({255: "boarder"})
                        class_lables.update({0: "nothing"})
                        wandb_iamge = wandb.Image(
                            cv2.resize(
                                np.moveaxis(
                                    np.squeeze(img.cpu().numpy(), axis=0), 0, -1
                                ),
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
                step_val += 1

                tbar.set_description("mIOU: %.2f" % (mIOU * 100.0))
        if args.use_wandb:
            wandb.log({"Pictures": wandb_iamges, "step_epoch": epoch})
            wandb.log({"final mIOU": mIOU})
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
            if args.dataset == "melanoma" or args.dataset == "breastCancer":
                mask = mask.clip(max=1)
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(
                    num_classes=21
                    if args.dataset == "pascal"
                    else 2
                    if args.dataset == "melanoma"
                    else 2
                    if args.dataset == "breastCancer"
                    else 19
                )
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    labeled_ids = []
    with open(args.split_file_path, "r") as file:
        split_dict = yaml.load(file, Loader=yaml.FullLoader)
        labeled_ids = split_dict[args.val_split]["labeled"]

    yaml_dict = dict()
    yaml_dict[args.val_split] = dict(
        labeled=labeled_ids,
        reliable=[i[0] for i in id_to_reliability[: len(id_to_reliability) // 2]],
        unreliable=[i[0] for i in id_to_reliability[len(id_to_reliability) // 2:]],
    )
    # save to yaml
    with open(
        os.path.join(args.reliable_id_path, "reliable_ids.yaml"), "w+"
    ) as outfile:
        yaml.dump(yaml_dict, outfile, default_flow_style=False)


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(
        num_classes=21
        if args.dataset == "pascal"
        else 2
        if args.dataset == "melanoma"
        else 2
        if args.dataset == "breastCancer"
        else 19
    )
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            if args.dataset == "melanoma" or args.dataset == "breastCancer":
                mask = mask.clip(max=1)  # clips max value to 1: 255 to 1
            img = img.cuda()
            pred = model(img, args.use_tta)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode="P")
            pred.putpalette(cmap)

            pred.save(
                "%s/%s" % (args.pseudo_mask_path, os.path.basename(id[0].split(" ")[1]))
            )

            tbar.set_description("mIOU: %.2f" % (mIOU * 100.0))


def test(model, dataloader, args):
    metric = mulitmetrics(
        num_classes=21
        if args.dataset == "pascal"
        else 2
        if args.dataset == "melanoma"
        else 2
        if args.dataset == "breastCancer"
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
            if args.dataset == "melanoma" or args.dataset == "breastCancer":
                mask = mask.clip(max=1)  # clips max value to 1: 255 to 1
            i = i + 1
            img = img.cuda()
            pred = model(img, args.use_tta)
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
        args.epochs = {
            "pascal": 80,
            "cityscapes": 240,
            "melanoma": 80,
            "breastCancer": 80,
        }[args.dataset]
    # if args.lr is None:
    #     args.lr = {'pascal': 0.001, 'cityscapes': 0.004, 'melanoma': 0.001, 'breastCancer': 0.001}[args.dataset] / 16 * args.batch_size
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
