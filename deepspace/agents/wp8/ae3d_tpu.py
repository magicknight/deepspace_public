'''
 ┌─────────────────────────────────────────────────────────────┐
 │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
 ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
 │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
 ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
 │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
 ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
 │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
 ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
 │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
 │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
 │      └───┴─────┴───────────────────────┴─────┴───┘          │
 └─────────────────────────────────────────────────────────────┘

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-06-30 19:41:16
LastEditors: Zhihua Liang
LastEditTime: 2021-06-30 19:41:26
FilePath: /home/zhihua/framework/deepspace/deepspace/agents/wp8/ae3d_tpu.py
'''


import numpy as np
from torch._C import device
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import lr_scheduler
from torch import nn

from tensorboardX.utils import make_grid
from tensorboardX import SummaryWriter
from torchinfo import summary

from deepspace.agents.base import BasicAgent
from deepspace.graphs.models.autoencoder.ae3d import Residual3DAE
from deepspace.datasets.wp8.wp8_npy_break import Loader
from deepspace.graphs.weights_initializer import xavier_weights
from deepspace.utils.metrics import AverageMeter
from deepspace.utils.data import save_npy
from deepspace.graphs.scheduler.scheduler import wrap_optimizer_with_scheduler

from commontools.setup import config

if config.deepspace.device == 'tpu':
    from torch_xla.core import xla_model
    from torch_xla.distributed import parallel_loader, xla_multiprocessing
    from torch_xla.test import test_utils


class Agent(BasicAgent):
    def __init__(self):
        super().__init__()

        # distribute: is master process?
        if config.deepspace.device == 'tpu':
            self.master = xla_model.is_master_ordinal()
        else:
            self.master = None

        # define models
        self.model = Residual3DAE(
            input_shape=config.deepspace.image_size,
            encoder_sizes=config.deepspace.encoder_sizes,
            fc_sizes=config.deepspace.fc_sizes,
            color_channels=config.deepspace.image_channels,
            temporal_strides=config.deepspace.temporal_strides
        )
        # model go to tpu
        self.model = self.model.to(self.device)

        self.model.apply(xavier_weights)

        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.deepspace.learning_rate if 'learning_rate' in config.deepspace else 1e-3,
        )

        # define data_loader
        # load dataset with parallel data loader
        self.data_loader = Loader()
        if config.deepspace.mode == 'train':
            self.train_loader = parallel_loader.MpDeviceLoader(self.data_loader.train_loader, self.device)
            self.train_iterations = self.data_loader.train_iterations
            self.valid_loader = parallel_loader.MpDeviceLoader(self.data_loader.valid_loader, self.device)
            self.valid_iterations = self.data_loader.valid_iterations

            # Define Scheduler
            self.scheduler = wrap_optimizer_with_scheduler(
                self.optimizer_gen,
                scheduler_type=config.deepspace.scheduler,
                scheduler_divisor=config.deepspace.scheduler_divisor,
                scheduler_divide_every_n_epochs=config.deepspace.scheduler_divide_every_n_epochs,
                num_steps_per_epoch=self.train_iterations,
                summary_writer=self.summary_writer
            )
        elif config.deepspace.mode == 'test':
            if config.common.distribute:
                self.test_loader = parallel_loader.MpDeviceLoader(self.data_loader.test_loader, self.device)
                self.test_iterations = self.data_loader.test_iterations // xla_model.xrt_world_size()
            else:
                # if not on distribution.
                self.test_loader = self.data_loader.test_loader
                self.test_iterations = self.data_loader.test_iterations

        # define loss
        self.loss = nn.MSELoss()
        # metric
        self.best_metric = 100

        if self.master:
            # Tensorboard Writer
            self.summary_writer = SummaryWriter(log_dir=config.swap.summary_dir, comment='wp8 gan3d npy on tpu')
            # add model to tensorboard and print parameter to screen
            summary(self.model, (1, config.deepspace.image_channels, *config.deepspace.image_size), device=self.device, depth=6)
            dummy_input = torch.randn(1, config.deepspace.image_channels, *config.deepspace.image_size).to(self.device)
            self.summary_writer.add_graph(self.model, (dummy_input), verbose=False)

        self.checkpoint = ['current_epoch', 'current_iteration', 'model.state_dict', 'optimizer.state_dict']

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(file_name=config.deepspace.checkpoint_file)

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, config.deepspace.max_epoch):
            xla_model.master_print('========= Epoch {} train begin {} =========='.format(epoch, test_utils.now()))
            self.current_epoch = epoch
            train_loss = self.train_one_epoch()
            xla_model.master_print('========= Epoch {} train end {} =========='.format(epoch, test_utils.now()))
            valid_loss = self.validate()
            self.scheduler.step()
            is_best = valid_loss < self.best_metric
            if valid_loss < self.best_metric:
                self.best_metric = valid_loss
            backup_checkpoint = False
            if self.current_epoch % config.deepspace.save_model_step == 0:
                backup_checkpoint = True
            self.save_checkpoint(config.deepspace.checkpoint_file,  is_best=is_best, backup_checkpoint=backup_checkpoint)

    def train_one_epoch(self):
        """
            One epoch of training
            :return:
            """
        # distribute tracker
        tracker = xla_model.RateTracker()
        # Initialize tqdm dataset
        tqdm_batch = tqdm(
            self.train_loader,
            total=self.train_iterations // xla_model.xrt_world_size(),
            desc="train on Epoch-{epoch}/{max_epoch}-".format(epoch=self.current_epoch, max_epoch=config.deepspace.max_epoch)
        )
        # Set the model to be in training mode (for batchnorm)
        self.model.train()
        # Initialize average meters
        epoch_loss = AverageMeter(device=self.device)
        # loop images
        for input_images, target_images in tqdm_batch:
            # for images, paths in self.train_loader:
            self.optimizer.zero_grad()
            # fake data---
            recon_images = self.model(input_images)
            # train the model now---
            loss = self.loss(recon_images, target_images)
            loss.backward()
            # self.optimizer.step()
            xla_model.optimizer_step(self.optimizer)
            epoch_loss.update(loss)
            self.current_iteration += 1
            tracker.add(config.deepspace.train_batch)
        # close dataloader
        tqdm_batch.close()
        # logging
        if self.master:
            self.summary_writer.add_scalar("epoch-training/loss", epoch_loss.val, self.current_epoch)
        xla_model.master_print("Training Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(epoch_loss.val) + " | ")
        xla_model.add_step_closure(self.train_update, args=(self.device, self.current_iteration, epoch_loss, tracker, self.current_epoch, self.summary_writer))
        return epoch_loss.val

    def validate(self):
        # distribute tracker
        tracker = xla_model.RateTracker()
        tqdm_batch = tqdm(
            self.valid_loader,
            total=self.valid_iterations // xla_model.xrt_world_size(),
            desc="validate at -{epoch}/{max_epoch}-".format(epoch=self.current_epoch, max_epoch=config.deepspace.max_epoch)
        )
        self.model.eval()
        epoch_loss = AverageMeter(device=self.device)
        # recon_images_sample = []
        # input_images_sample = []
        with torch.no_grad():
            for input_images, target_images in tqdm_batch:
                # fake data---
                recon_images = self.model(input_images)
                loss = self.loss(recon_images, target_images)
                epoch_loss.update(loss)

                # save the reconstructed image
                # if index >= config.deepspace.save_images[0] and index < config.deepspace.save_images[1]:
                #     recon_images = recon_images.view(recon_images.shape[0], recon_images.shape[1], int(recon_images.shape[2] * recon_images.shape[3] / 4), -1)
                #     images = images.view(images.shape[0], images.shape[1], int(images.shape[2] * images.shape[3] / 4), -1)
                #     recon_images = recon_images.squeeze().detach().cpu().numpy()
                #     recon_images_sample.append(np.expand_dims(recon_images, axis=1))
                #     images = images.squeeze().detach().cpu().numpy()
                #     input_images_sample.append(np.expand_dims(images, axis=1))
                tracker.add(config.deepspace.validate_batch)

            tqdm_batch.close()
            # recon_images_sample = np.concatenate(recon_images_sample)
            # input_images_sample = np.concatenate(input_images_sample)
            # recon_canvas = make_grid(recon_images_sample)
            # input_canvas = make_grid(input_images_sample)
            if self.master:
                self.summary_writer.add_scalar("epoch-validation/loss", epoch_loss.val, self.current_epoch)
                # self.summary_writer.add_image('Recon_images', recon_canvas, self.current_epoch)
                # self.summary_writer.add_image('Input_images', input_canvas, self.current_epoch)
            # logging
            xla_model.master_print("validate Results at epoch-" + str(self.current_epoch) + " | " + ' epoch_loss: ' + str(epoch_loss.val) + " | ")
            xla_model.add_step_closure(self.test_update, args=(self.device, self.current_iteration, epoch_loss, self.current_epoch))
            return epoch_loss.val

    def test(self):
        test_root = Path(config.deepspace.test_output_dir)
        # tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations, desc="test at -{}-".format(self.current_epoch))
        tqdm_batch = tqdm(
            self.test_loader,
            total=self.test_iterations,
            # desc="test at -{epoch}- on device- {device}/{ordinal}".format(epoch=self.current_epoch, device=self.device, ordinal=xla_model.get_ordinal())
            desc="test at -{epoch}/{max_epoch}-".format(epoch=self.current_epoch, max_epoch=config.deepspace.max_epoch)
        )
        self.model.eval()
        epoch_loss = AverageMeter(device=self.device)
        recon_images_sample = []
        input_images_sample = []
        image_paths = []
        with torch.no_grad():
            for images, paths in tqdm_batch:
                # fake data---
                recon_images = self.model(images)
                # train the model now---
                loss = self.loss(recon_images, images)
                epoch_loss.update(loss)

                # save the reconstructed image
                recon_images_sample.append(recon_images)
                input_images_sample.append(images)
                image_paths.append(paths)

        tqdm_batch.close()

        image_paths = np.concatenate(np.array(image_paths))
        recon_images_sample = torch.cat(recon_images_sample).squeeze().detach().cpu().numpy()
        input_images_sample = torch.cat(input_images_sample).squeeze().detach().cpu().numpy()

        recon_paths = [test_root / ('recon' + '_' + Path(image_paths[index]).name) for index in range(recon_images_sample.shape[0])]
        input_paths = [test_root / ('input' + '_' + Path(image_paths[index]).name) for index in range(input_images_sample.shape[0])]
        save_npy(recon_images_sample, paths=recon_paths)
        save_npy(input_images_sample, paths=input_paths)

        # logging
        if config.deepspace.device == 'tpu':
            xla_model.master_print("test Results at epoch-" + str(self.current_epoch) + " | " + ' epoch_loss: ' + str(epoch_loss.val) + " | ")
        else:
            print(("test Results at epoch-" + str(self.current_epoch) + " | " + ' epoch_loss: ' + str(epoch_loss.val) + " | "))
        return epoch_loss.val

    def train_update(self, device, step, loss, tracker, epoch, writer):
        test_utils.print_training_update(
            device,
            step,
            loss.val,
            tracker.rate(),
            tracker.global_rate(),
            epoch,
            summary_writer=writer)

    def test_update(self, device, step, loss, epoch):
        test_utils.print_test_update(
            device,
            1.0 - loss.val,
            epoch,
            step)
