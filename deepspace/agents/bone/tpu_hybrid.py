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
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD, lr_scheduler
from torch import nn


from tensorboardX.utils import make_grid
from tensorboardX import SummaryWriter
from torchinfo import summary

from deepspace.agents.base import BasicAgent
from deepspace.graphs.models.transformer.hybrid import Hybrid
from deepspace.datasets.voxel.npy_2d_ml21 import Loader
from deepspace.utils.metrics import AverageMeter
from deepspace.utils.data import save_npy
from deepspace.graphs.weights_initializer import xavier_weights

from commontools.setup import config

if config.deepspace.device == 'tpu':
    from torch_xla.core import xla_model
    from torch_xla.distributed import parallel_loader, xla_multiprocessing
    from torch_xla.test import test_utils


class Agent(BasicAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # distribute: is master process?
        if config.deepspace.device == 'tpu':
            self.master = xla_model.is_master_ordinal()
        else:
            self.master = None

        # define models
        self.model = Hybrid(
            image_size=config.deepspace.image_size,
            patch_size=config.deepspace.patch_size,
            dim=config.deepspace.dim,
            depth=config.deepspace.depth,
            heads=config.deepspace.heads,
            in_channels=config.deepspace.image_channels,
            out_channels=config.deepspace.out_channels,
            dim_head=config.deepspace.dim_head,
            dropout=config.deepspace.dropout,
            emb_dropout=config.deepspace.emb_dropout,
            scale_dim=config.deepspace.scale_dim,
        )
        # model go to tpu
        self.model = self.model.to(self.device)
        # initialize weights
        self.model.apply(xavier_weights)

        # Create instance from the optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.deepspace.learning_rate,
            weight_decay=config.deepspace.weight_decay
        )
        # self.optimizer = SGD(
        #     self.model.parameters(),
        #     lr=config.deepspace.learning_rate,
        #     momentum=config.deepspace.momentum,
        #     weight_decay=config.deepspace.weight_decay,
        # )
        # define data_loader
        # load dataset with parallel data loader
        self.data_loader = Loader(shared_array=self.shared_array)
        if config.deepspace.mode == 'train':
            self.train_loader = parallel_loader.MpDeviceLoader(self.data_loader.train_loader, self.device)
            self.train_iterations = self.data_loader.train_iterations
            self.valid_loader = parallel_loader.MpDeviceLoader(self.data_loader.valid_loader, self.device)
            self.valid_iterations = self.data_loader.valid_iterations

            # self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config.deepspace.t_initial, T_mult=config.deepspace.t_mult)
        elif config.deepspace.mode == 'test':
            if config.common.distribute:
                self.test_loader = parallel_loader.MpDeviceLoader(self.data_loader.test_loader, self.device)
                self.test_iterations = self.data_loader.test_iterations // xla_model.xrt_world_size()
            else:
                # if not on distribution.
                self.test_loader = self.data_loader.test_loader
                self.test_iterations = self.data_loader.test_iterations

        # Define Scheduler
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config.deepspace.T_0, T_mult=config.deepspace.T_mult, )

        # define loss
        # self.loss = nn.L1Loss()
        self.loss = nn.MSELoss()
        # self.recon_loss = SSIM_Loss(data_range=1, channel=config.deepspace.image_channels, win_size=config.deepspace.ssim_win_size)
        # self.criterion = SSIM(data_range=1, channel=config.deepspace.image_channels, win_size=config.deepspace.ssim_win_size)
        # self.recon_loss = nn.L1Loss()
        # self.criterion = nn.L1Loss()
        # metric
        self.best_metric = 100

        if self.master:
            # Tensorboard Writer
            self.summary_writer = SummaryWriter(log_dir=config.swap.summary_dir, comment=config.summary.description)
            # add model to tensorboard and print parameter to screen
            summary(self.model, (1, config.deepspace.image_channels, config.deepspace.image_size, config.deepspace.image_size), device=self.device, depth=6)
            dummy_input = torch.randn(1, config.deepspace.image_channels, config.deepspace.image_size, config.deepspace.image_size,).to(self.device)
            self.summary_writer.add_graph(self.model, (dummy_input), verbose=False)

        self.checkpoint = ['current_epoch', 'current_iteration', 'model.state_dict', 'optimizer.state_dict', 'scheduler.state_dict']
        # self.checkpoint = ['model.state_dict'] # only for transition from pre-trained model to fine-tuned model
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
            if 'validate_step' in config.deepspace and self.current_epoch % config.deepspace.validate_step != 0:
                continue
            valid_loss = self.validate()
            self.scheduler.step(self.current_epoch)
            is_best = valid_loss < self.best_metric
            if valid_loss < self.best_metric:
                self.best_metric = valid_loss
            backup_checkpoint = False
            if self.current_epoch % config.deepspace.backup_model_step == 0:
                backup_checkpoint = True
            if self.current_epoch % config.deepspace.save_model_step == 0:
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
            # input_images.requires_grad = True
            # target_images.requires_grad = True
            # for images, paths in self.train_loader:
            self.optimizer.zero_grad()
            recon_images = self.model(input_images)
            # loss
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
        xla_model.master_print(
            "Training Results at epoch-" + str(self.current_epoch) + " | "
            + "loss: " + str(epoch_loss.val) + " | "
        )
        xla_model.add_step_closure(self.train_update, args=(self.device, self.current_iteration, epoch_loss, tracker, self.current_epoch, self.summary_writer))
        return epoch_loss.val

    @torch.no_grad()
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
        recon_images_sample = []
        target_images_sample = []
        input_images_sample = []
        index = 0
        for input_images, target_images in tqdm_batch:
            # fake data---
            recon_images = self.model(input_images)
            loss = self.loss(recon_images, target_images)
            epoch_loss.update(loss)
            tracker.add(config.deepspace.validate_batch)
            # save the reconstructed image
            if self.master and self.current_epoch % config.deepspace.save_image_step == 0:
                if index >= config.deepspace.save_images[0] and index < config.deepspace.save_images[1]:
                    recon_images_sample.append(recon_images)
                    target_images_sample.append(target_images)
                    input_images_sample.append(input_images)
            index = index + 1

        tqdm_batch.close()
        if self.master:
            self.summary_writer.add_scalar("epoch-validation/loss", epoch_loss.val, self.current_epoch)
            # save reconstruct image to tensorboard
            if self.current_epoch % config.deepspace.save_image_step == 0:
                recon_images_sample = torch.cat(recon_images_sample, dim=0).detach().cpu().numpy()
                target_images_sample = torch.cat(target_images_sample).detach().cpu().numpy()
                input_images_sample = torch.cat(input_images_sample).detach().cpu().numpy()
                recon_canvas = make_grid(recon_images_sample)
                self.summary_writer.add_image('recon_images', recon_canvas, self.current_epoch)
                input_canvas = make_grid(target_images_sample)
                self.summary_writer.add_image('target_images', input_canvas, self.current_epoch)
                defect_canvas = make_grid(input_images_sample[:, config.deepspace.image_channels//2 - 1:config.deepspace.image_channels//2 + 2, :, :])
                self.summary_writer.add_image('input_images', defect_canvas, self.current_epoch)
        # logging
        xla_model.master_print("validate Results at epoch-" + str(self.current_epoch) + " | " + ' epoch_loss: ' + str(epoch_loss.val) + " | ")
        xla_model.add_step_closure(self.test_update, args=(self.device, self.current_iteration, epoch_loss, self.current_epoch))
        return epoch_loss.val

    @torch.no_grad()
    def test(self):
        # tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations, desc="test at -{}-".format(self.current_epoch))
        tqdm_batch = tqdm(
            self.test_loader,
            total=self.test_iterations,
            # desc="test at -{epoch}- on device- {device}/{ordinal}".format(epoch=self.current_epoch, device=self.device, ordinal=xla_model.get_ordinal())
            desc="test at -{epoch}/{max_epoch}-".format(epoch=self.current_epoch, max_epoch=config.deepspace.max_epoch)
        )
        self.model.eval()
        epoch_loss = AverageMeter(device=self.device)
        recon_data = np.zeros(self.data_loader.data_shape, dtype=np.float32)
        size = config.deepspace.image_size
        for images, coors in tqdm_batch:
            # fake data---
            rot1_p, contrastive1_p, recon_images, r_w, cn_w, rec_w = self.model(images)
            loss = self.criterion(recon_images, images)
            epoch_loss.update(loss)
            for recon, coor in zip(recon_images, coors):
                recon_data[coor[0]:coor[0]+size[0], coor[1]:coor[1]+size[1], coor[2]:coor[2]+size[2]] = recon.squeeze().detach().cpu().numpy()
        tqdm_batch.close()
        # shape = config.deepspace.data_shape
        # recon_data = recon_data[0:shape[0], 0:shape[1], 0:shape[2]]
        # input_coors = [test_root / ('input' + '_' + Path(image_coors[index]).name) for index in range(input_images_sample.shape[0])]
        save_npy([recon_data], [config.deepspace.test_output])

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
            (100.0 - loss.val)/100.0,
            epoch,
            step)

    @torch.no_grad()
    def test_validate(self):
        # recon_data = np.zeros(self.data_loader.data_shape, dtype=np.float32)
        # distribute tracker
        tracker = xla_model.RateTracker()
        tqdm_batch = tqdm(
            self.valid_loader,
            total=self.valid_iterations // xla_model.xrt_world_size(),
            desc="validate at -{epoch}/{max_epoch}-".format(epoch=self.current_epoch, max_epoch=config.deepspace.max_epoch)
        )
        self.model.eval()
        epoch_loss = AverageMeter(device=self.device)
        recon_data_sample = []
        coor_sample = []
        for train_data, target_data, coors in tqdm_batch:
            # fake data---
            rot1_p, contrastive1_p, recon_data, r_w, cn_w, rec_w = self.model(train_data)
            recon_data_sample.append(recon_data)
            coor_sample.append(coors)
            loss = self.recon_loss(recon_data, target_data)
            epoch_loss.update(loss)
            tracker.add(config.deepspace.validate_batch)
        tqdm_batch.close()
        size = config.deepspace.image_size
        recon_array = self.shared_array[2]
        for recon, coor in zip(recon_data_sample, coor_sample):
            coor = coor.squeeze().detach().cpu().numpy().astype(np.int)
            recon = recon.squeeze().detach().cpu().numpy()
            for index in range(recon.shape[0]):
                recon_array[coor[index][0]:coor[index][0]+size[0], coor[index][1]:coor[index][1]+size[1], coor[index][2]:coor[index][2]+size[2]] = recon[index]
        # logging
        xla_model.master_print("validate Results at epoch-" + str(self.current_epoch) + " | " + ' epoch_loss: ' + str(epoch_loss.val) + " | ")
        xla_model.add_step_closure(self.test_update, args=(self.device, self.current_iteration, epoch_loss, self.current_epoch))
        if self.master:
            self.summary_writer.add_scalar("epoch-validation/loss", epoch_loss.val, self.current_epoch)
            for index in config.deepspace.save_images:
                self.summary_writer.add_image('Recon_images' + str(index), np.expand_dims(recon_array[index], axis=0), self.current_epoch)
        return epoch_loss.val
