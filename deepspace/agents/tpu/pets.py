
import shutil
from ssl import ALERT_DESCRIPTION_UNSUPPORTED_CERTIFICATE
import numpy as np
import json
from pprint import pprint
from tqdm import tqdm
from pathlib import Path
import deepdish as dd

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch import nn
from tensorboardX import SummaryWriter
from torchsummary import summary
from accelerate import Accelerator
from timm import create_model

from deepspace.agents.base import BasicAgent
from deepspace.datasets.public.pets import get_data_loader
from deepspace.utils.dirs import create_dirs
from deepspace.utils.data import save_images, make_heatmaps, make_masks
from deepspace.utils.metrics import evaluate_decision

from commontools.data.data import to_uint8
from commontools.setup import config, logger

accelerator = Accelerator(fp16=config.deepspace.fp16, cpu=config.deepspace.cpu)


class PetAgent(BasicAgent):

    def __init__(self):
        super().__init__()
        # Initialize accelerator

        # dataloader
        self.train_loader, self.valid_loader, self.num_classes = get_data_loader()
        # define models (policy and target)
        # Instantiate the model (we build the model here so that the seed also control new weights initialization)
        self.model = create_model("resnet50d", pretrained=True, num_classes=self.num_classes)
        # Freezing the base model
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True

        # We normalize the batches of images to be a bit faster.
        self.mean = torch.tensor(self.model.default_cfg["mean"])[None, :, None, None]
        self.std = torch.tensor(self.model.default_cfg["std"])[None, :, None, None]

        # define loss
        # Instantiate optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=config.deepspace.learning_rate)

        # Prepare everything
        # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the prepare method.
        self.model, self.optimizer, self.train_loader, self.valid_loader, self.mean, self.std = accelerator.prepare(self.model, self.optimizer, self.train_loader, self.valid_loader, self.mean, self.std)

        # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
        # may change its length.
        self.lr_scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=config.deepspace.learning_rate, epochs=config.deepspace.max_epoch, steps_per_epoch=len(self.train_loader))

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint()
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=config.swap.summary_dir, comment='Pet')

        # save checkpoint
        self.checkpoint = {
            'current_episode': self.current_episode,
            'current_iteration': self.current_iteration,
            'model.state_dict': self.model.state_dict(),
            'optimizer.state_dict': self.optimizer.state_dict(),
        }

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in tqdm(range(self.current_epoch, config.deepspace.max_epoch), desc="traing at -{}-, total epoches -{}-".format(self.current_epoch, config.deepspace.max_epoch)):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint(config.deepspace.checkpoint_file,  is_best=False)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        for step, batch in enumerate(self.train_loader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            inputs = (batch["image"] - self.mean) / self.std
            outputs = self.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
            accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

    def validate(self):
        self.model.eval()
        accurate = 0
        num_elems = 0
        for step, batch in enumerate(self.valid_loader):
            inputs = (batch["image"] - self.mean) / self.std
            with torch.no_grad():
                outputs = self.model(inputs)
            predictions = outputs.argmax(dim=-1)
            accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
            num_elems += accurate_preds.shape[0]
            accurate += accurate_preds.long().sum()

        eval_metric = accurate.item() / num_elems
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {self.current_epoch}: {100 * eval_metric:.2f}")
