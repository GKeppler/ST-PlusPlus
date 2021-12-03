from model.backbone.resnet import resnet50, resnet101, resnet18

from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from utils import count_params, meanIOU, color_map
import torch
from statistics import mean
import os


class BaseNet(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseNet")
        #initial learing rate
        parser.add_argument("--lr", type=float, default=0.001)
        return parent_parser

    def __init__(self, backbone, *args, **kwargs):
        lr = kwargs.get('lr')
        super(BaseNet, self).__init__()
        backbone_zoo = {'resnet18': resnet18, 'resnet50': resnet50, 'resnet101': resnet101}
        self.backbone_name = backbone
        self.backbone = backbone_zoo[backbone](pretrained=True)
        self.metric = meanIOU(num_classes=21) # change for dataset
        self.criterion = CrossEntropyLoss(ignore_index=255)
        self.previous_best = 0.0
        self.args = kwargs

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
            scales = [0.5, 0.75, 1.0, 1.5, 2.0]

            final_result = None

            for scale in scales:
                cur_h, cur_w = int(h * scale), int(w * scale)
                cur_x = F.interpolate(x, size=(cur_h, cur_w), mode='bilinear', align_corners=True)

                out = F.softmax(self.base_forward(cur_x), dim=1)
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                final_result = out if final_result is None else (final_result + out)

                out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3)
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                final_result += out

            return final_result

    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred = self(img)
        loss = self.criterion(pred, mask)
        #loss = F.cross_entropy(pred, mask, ignore_index=255)
        #loss.requires_grad = True
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred = self(img)
        self.metric.add_batch(torch.argmax(pred, dim=1).cpu().numpy(), mask.cpu().numpy())
        val_loss = self.metric.evaluate()[-1]
        # wandb.log({"mIOU": mIOU,"step_val":step_val})
        return{"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = outputs[-1]['val_loss']
        log = {'mean mIOU': val_loss * 100}
        mIOU = val_loss * 100.0
        if mIOU > self.previous_best:
            if self.previous_best != 0:
                os.remove(os.path.join(self.args['save_path'], '%s_%s_mIOU%.2f.pth' % (self.args["model"], self.backbone_name, self.previous_best)))
            self.previous_best = mIOU
            torch.save(self.state_dict(), os.path.join(self.args['save_path'], '%s_%s_mIOU%.2f.pth' % (self.args["model"], self.backbone_name, mIOU)))
        return{"log": log, "val_loss": val_loss}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        img, mask, id = batch
        pred = self(img)
        #batch_size = batch[0].size(0)
        #prediction_file = getattr(self, "prediction_file", "predictions.pt")
        #lazy_ids = torch.arange(batch_idx * batch_size, batch_idx * batch_size + batch_size)
        #self.write_prediction("idxs", lazy_ids, prediction_file)
        # self.write_prediction("preds", output, prediction_file)
        return [pred, mask, id]

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=0.01,#self.lr,
            momentum=0.9,
            weight_decay=1e-4
            )
        #scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min')
        return [optimizer]#, [scheduler] 