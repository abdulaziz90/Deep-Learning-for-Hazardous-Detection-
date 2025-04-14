import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
# from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import numpy as np

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        # real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        # results = self.forward(real_img)
        train_loss = self.model.loss_function(*results,
                                              inputs=batch,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        # real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        # results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            inputs=batch,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        optims = []
        scheds = []

        # First optimizer
        optimizer = optim.Adam(self.model.parameters(),
                            lr=self.params.get('LR', 1e-3),
                            weight_decay=self.params.get('weight_decay', 0))
        optims.append(optimizer)

        # Second optimizer (if needed)
        if 'LR_2' in self.params and self.params['submodel']:
            optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                    lr=self.params['LR_2'])
            optims.append(optimizer2)

        # First scheduler (if needed)
        if 'scheduler_gamma' in self.params:
            scheduler = {
                'scheduler': optim.lr_scheduler.ExponentialLR(optims[0], gamma=self.params['scheduler_gamma']),
                'interval': 'epoch',  # or 'step'
                # 'monitor': 'val_loss',  # if you want to monitor some val metric
            }
            scheds.append(scheduler)

        # Second scheduler (if needed)
        if 'scheduler_gamma_2' in self.params and len(optims) > 1:
            scheduler2 = {
                'scheduler': optim.lr_scheduler.ExponentialLR(optims[1], gamma=self.params['scheduler_gamma_2']),
                'interval': 'epoch',  # or 'step'
                # 'monitor': 'val_loss',  # if you want to monitor some val metric
            }
            scheds.append(scheduler2)

        return optims, scheds if scheds else None

    
    def test_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        test_loss = self.model.loss_function(*results,
                                            inputs=batch,
                                            M_N=1.0,
                                            optimizer_idx=0,
                                            batch_idx=batch_idx)

        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True)

        return test_loss


    def predict_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)

        return results

