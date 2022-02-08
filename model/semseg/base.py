from model.backbone.resnet import resnet50, resnet101, resnet18

import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from utils import meanIOU, color_map
import torch
import os
from PIL import Image
import numpy as np


class BaseNet(pl.LightningModule):
    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("BaseNet")
    #     # initial learing rate
    #     # parser.add_argument("--lr", type=float, default=0.001)
    #     return parent_parser

    def __init__(self, backbone, nclass, args):
        super(BaseNet, self).__init__()
        backbone_zoo = {
            "resnet18": resnet18,
            "resnet50": resnet50,
            "resnet101": resnet101,
        }
        self.backbone_name = backbone
        if backbone is not None:
            self.backbone = backbone_zoo[backbone](pretrained=True)
        self.metric = meanIOU(num_classes=nclass)
        self.predict_metric = meanIOU(num_classes=nclass)
        self.criterion = CrossEntropyLoss()  # ignore_index=255)
        self.previous_best = 0.0
        self.args = args

    def base_forward(self, x):
        h, w = x.shape[-2:]
        x = self.backbone.base_forward(x)[-1]
        x = self.head(x)
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True)
        return x

    def forward(self, x, tta=False):
        if not tta:
            return self.base_forward(x)

        else:
            h, w = x.shape[-2:]
            # scales = [0.5, 0.75, 1.0]
            # to avoid cuda out of memory
            scales = [0.5, 0.75, 1.0, 1.5, 2.0]

            final_result = None

            for scale in scales:
                cur_h, cur_w = int(h * scale), int(w * scale)
                cur_x = F.interpolate(
                    x, size=(cur_h, cur_w), mode="bilinear", align_corners=True
                )

                out = F.softmax(self.base_forward(cur_x), dim=1)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                final_result = out if final_result is None else (final_result + out)

                out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                final_result += out

            return final_result

    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred = self(img)
        if self.args.dataset == "melanoma":
            mask = mask.clip(max=1)  # clips max value to 1: 255 to 1
        loss = self.criterion(pred, mask)
        # loss = F.cross_entropy(pred, mask, ignore_index=255)
        # loss.requires_grad = True
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        img, mask, id = batch
        pred = self(img)
        self.metric.add_batch(
            torch.argmax(pred, dim=1).cpu().numpy(), mask.cpu().numpy()
        )
        val_acc = self.metric.evaluate()[-1]
        # wandb.log({"mIOU": mIOU,"step_val":step_val})
        return {"val_acc": val_acc}

    def validation_epoch_end(self, outputs):
        val_acc = outputs[-1]["val_acc"]
        log = {"mean mIOU": val_acc * 100}
        mIOU = val_acc * 100.0
        if mIOU > self.previous_best:
            if self.previous_best != 0:
                os.remove(
                    os.path.join(
                        self.args.save_path,
                        "%s_%s_mIOU%.2f.pth"
                        % (self.args.model, self.backbone_name, self.previous_best),
                    )
                )
            self.previous_best = mIOU
            torch.save(
                self.state_dict(),
                os.path.join(
                    self.args.save_path,
                    "%s_%s_mIOU%.2f.pth" % (self.args.model, self.backbone_name, mIOU),
                ),
            )
        return {"log": log, "val_acc": val_acc}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        img, mask, id = batch
        pred = self(img)
        pred = torch.argmax(pred, dim=1).cpu()

        # for metric checking progressbar callback not implemented
        # self.predict_metric.add_batch(pred.numpy(), mask.cpu().numpy())
        # mIOU = self.predict_metric.evaluate()[-1]
        pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode="P")
        pred.putpalette(color_map(self.args.dataset))
        pred.save(
            "%s/%s"
            % (self.args.pseudo_mask_path, os.path.basename(id[0].split(" ")[1]))
        )
        return [pred, mask, id]

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4  # self.lr,
        )
        # scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min')
        return [optimizer]  # , [scheduler]
