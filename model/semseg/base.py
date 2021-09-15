from model.backbone.resnet import resnet50, resnet101

from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from utils import count_params, meanIOU, color_map
import torch
from statistics import mean

class BaseNet(pl.LightningModule):
    def __init__(self, backbone):
        super(BaseNet, self).__init__()
        backbone_zoo = {'resnet50': resnet50, 'resnet101': resnet101}
        self.backbone = backbone_zoo[backbone](pretrained=True)

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
            pred, mask = batch
            loss = F.cross_entropy(pred, mask, ignore_index=255)
            loss.requires_grad = True
            return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        pred, mask, path = batch
        metric = meanIOU(num_classes=21) # change for dataset
        metric.add_batch(torch.argmax(pred, dim=1).numpy(), mask.numpy())
        val_loss = metric.evaluate()[-1]
        # wandb.log({"mIOU": mIOU,"step_val":step_val})
        return{"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = mean([x['val_loss'] for x in outputs])
        log = {'avg_val_loss': val_loss}
        # "val_loss" saves checkpoints: lock for autocheckpoints
        return{"log": log, "val_loss": val_loss}

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        img, mask, id = batch
        return self(img)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
