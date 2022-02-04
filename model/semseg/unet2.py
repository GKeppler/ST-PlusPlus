from model.semseg.base import BaseNet
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import SGD
import pytorch_lightning as pl

#this is a customized uent from https://github.com/milesial/Pytorch-UNet
#it uses padding such that the input shape is the same as the output.
# additional batchnormalization





class Unet2(pl.LightningModule):
    def __init__(self, backbone, nclass, args=None):
        super(Unet2, self).__init__()
        enc_chs=(3,64,128,256,512,1024)
        dec_chs=(1024, 512, 256, 128, 64)
        self.out_sz=(572,572)
        retain_dim=False,
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], nclass, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out

    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred = self(img)
        loss = self.criterion(pred, mask)
        #loss = F.cross_entropy(pred, mask, ignore_index=255)
        #loss.requires_grad = True
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        img, mask, id = batch
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
    

class Block(pl.LightningModule):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(pl.LightningModule):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(pl.LightningModule):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
