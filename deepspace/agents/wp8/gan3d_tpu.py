
import numpy as np
from torch._C import device
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import lr_scheduler
from torch import nn

from torch_xla.core import xla_model
from torch_xla.distributed import parallel_loader, xla_multiprocessing
from torch_xla.test import test_utils

from tensorboardX.utils import make_grid
from tensorboardX import SummaryWriter
from torchinfo import summary

from deepspace.agents.base import BasicAgent
from deepspace.graphs.models.autoencoder.ae3d import Residual3DAE
from deepspace.datasets.wp8.wp8_npy import NPYDataLoader
from deepspace.graphs.weights_initializer import xavier_weights
from deepspace.utils.metrics import AverageMeter
from deepspace.utils.data import save_npy

from commontools.setup import config


class WP8Agent(BasicAgent):
    def __init__(self):
        super().__init__()

        # distribute: is master process?
        self.master = xla_model.is_master_ordinal()

        # define models
        self.model = Residual3DAE(
            input_shape=config.deepspace.image_size,
            encoder_sizes=config.deepspace.encoder_sizes,
            fc_sizes=config.deepspace.fc_sizes,
            color_channels=config.deepspace.image_channels,
            temporal_strides=config.deepspace.temporal_strides
        )
        # logger.info('========== device is =================== {device}, on {master}'.format(device=self.device, master='master' if self.master else 'worker'))
        # model go to tpu
        self.model = self.model.to(self.device)

        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.deepspace.learning_rate if 'learning_rate' in config.deepspace else 1e-3,
        )

        # Define Scheduler
        def lambda1(epoch): return pow(1 - epoch / config.deepspace.max_epoch, 0.9)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        self.model.apply(xavier_weights)

        # define data_loader
        # load dataset with parallel data loader
        self.data_loader = NPYDataLoader()
        self.train_loader = parallel_loader.MpDeviceLoader(self.data_loader.train_loader, self.device)
        self.train_iterations = self.data_loader.train_iterations
        self.valid_loader = parallel_loader.MpDeviceLoader(self.data_loader.valid_loader, self.device)
        self.valid_iterations = self.data_loader.valid_iterations
        self.test_loader = parallel_loader.MpDeviceLoader(self.data_loader.test_loader, self.device)
        self.test_iterations = self.data_loader.test_iterations

        # define loss
        self.loss = nn.MSELoss()
        # self.loss = self.loss.to(self.device)
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
        # for epoch in tqdm(range(self.current_epoch, config.deepspace.max_epoch), desc="traing at -{}-, total epoches -{}-".format(self.current_epoch, config.deepspace.max_epoch)):
        for epoch in range(self.current_epoch, config.deepspace.max_epoch):
            xla_model.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
            self.current_epoch = epoch
            train_loss = self.train_one_epoch()
            xla_model.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
            valid_loss = self.validate()
            self.scheduler.step()
            is_best = valid_loss < self.best_metric
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
            total=self.train_iterations,
            desc="train on Epoch-{epoch}- on device- {device}/{ordinal}".format(epoch=self.current_epoch, device=self.device, ordinal=xla_model.get_ordinal()))
        # Set the model to be in training mode (for batchnorm)
        self.model.train()
        # Initialize average meters
        epoch_loss = AverageMeter()
        # loop images
        for images, paths in tqdm_batch:
            # for images, paths in self.train_loader:
            # images = images.to(self.device, dtype=torch.float32)
            images.requires_grad = True
            self.optimizer.zero_grad()
            # fake data---
            recon_images = self.model(images)
            # train the model now---
            loss = self.loss(recon_images, images)
            loss.backward()
            # self.optimizer.step()
            xla_model.optimizer_step(self.optimizer)
            epoch_loss.update(loss.item())
            self.current_iteration += 1
            tracker.add(config.deepspace.train_batch)
        # close dataloader
        # tqdm_batch.close()
        # logging
        if self.master:
            self.summary_writer.add_scalar("epoch-training/loss", epoch_loss.val, self.current_epoch)
        xla_model.master_print("Training Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(epoch_loss.val) + " | ")
        xla_model.add_step_closure(self.train_update, args=(self.device, self.current_iteration, epoch_loss, tracker, self.current_epoch, self.summary_writer))
        # save images
        # save the reconstructed image
        return epoch_loss.val

    def validate(self):
        # distribute tracker
        tracker = xla_model.RateTracker()
        tqdm_batch = tqdm(
            self.valid_loader,
            total=self.valid_iterations,
            desc="validate at -{epoch}- on device- {device}/{ordinal}".format(epoch=self.current_epoch, device=self.device, ordinal=xla_model.get_ordinal()))
        self.model.eval()
        epoch_loss = AverageMeter()
        recon_images_sample = []
        input_images_sample = []
        index = 0
        with torch.no_grad():
            for images, paths in tqdm_batch:
                images = images.to(self.device, dtype=torch.float32)
                # fake data---
                recon_images = self.model(images)
                # train the model now---
                loss = self.loss(recon_images, images)
                epoch_loss.update(loss.item())

                # save the reconstructed image
                if index >= config.deepspace.save_images[0] and index < config.deepspace.save_images[1]:
                    recon_images = recon_images.view(recon_images.shape[0], recon_images.shape[1], int(recon_images.shape[2] * recon_images.shape[3] / 4), -1)
                    images = images.view(images.shape[0], images.shape[1], int(images.shape[2] * images.shape[3] / 4), -1)
                    recon_images = recon_images.squeeze().detach().cpu().numpy()
                    recon_images_sample.append(np.expand_dims(recon_images, axis=1))
                    images = images.squeeze().detach().cpu().numpy()
                    input_images_sample.append(np.expand_dims(images, axis=1))
                index = index + 1
                tracker.add(config.deepspace.validate_batch)

            recon_images_sample = np.concatenate(recon_images_sample)
            input_images_sample = np.concatenate(input_images_sample)
            recon_canvas = make_grid(recon_images_sample)
            input_canvas = make_grid(input_images_sample)
            if self.master:
                self.summary_writer.add_scalar("epoch-validation/loss", epoch_loss.val, self.current_epoch)
                self.summary_writer.add_image('Recon_images', recon_canvas, self.current_epoch)
                self.summary_writer.add_image('Input_images', input_canvas, self.current_epoch)
            # logging
            xla_model.master_print("validate Results at epoch-" + str(self.current_epoch) + " | " + ' epoch_loss: ' + str(epoch_loss.val) + " | ")
            xla_model.add_step_closure(self.test_update, args=(self.device, self.current_iteration, epoch_loss, self.current_epoch))
            tqdm_batch.close()
            return epoch_loss.val

    def test(self):
        test_root = Path(config.deepspace.test_output_dir)
        tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations, desc="test at -{}-".format(self.current_epoch))
        self.model.eval()
        epoch_loss = AverageMeter()
        recon_images_sample = []
        input_images_sample = []
        image_paths = []
        index = 0
        with torch.no_grad():
            for images, paths in tqdm_batch:
                images = images.to(self.device, dtype=torch.float32)
                # fake data---
                recon_images = self.model(images)
                # train the model now---
                loss = self.loss(recon_images, images)
                epoch_loss.update(loss.item())

                # save the reconstructed image
                recon_images = recon_images.squeeze().detach().cpu().numpy()
                recon_images_sample.append(recon_images)
                images = images.squeeze().detach().cpu().numpy()
                input_images_sample.append(images)
                for each_path in paths:
                    image_paths.append(Path(each_path).name)
                index = index + 1

            recon_images_sample = np.concatenate(recon_images_sample)
            input_images_sample = np.concatenate(input_images_sample)

            # recon_paths = [test_root / ('recon' + '_' + str(index) + '.' + config.deepspace.data_format) for index in range(recon_images_sample.shape[0])]
            # input_paths = [test_root / ('input' + '_' + str(index) + '.' + config.deepspace.data_format) for index in range(input_images_sample.shape[0])]
            recon_paths = [test_root / ('recon' + '_' + image_paths[index]) for index in range(recon_images_sample.shape[0])]
            input_paths = [test_root / ('input' + '_' + image_paths[index]) for index in range(input_images_sample.shape[0])]
            save_npy(recon_images_sample, paths=recon_paths)
            save_npy(input_images_sample, paths=input_paths)

            # logging
            xla_model.master_print("test Results at epoch-" + str(self.current_epoch) + " | " + ' epoch_loss: ' + str(epoch_loss.val) + " | ")
            tqdm_batch.close()
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
