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
Date: 2021-06-25 09:35:50
LastEditors: Zhihua Liang
LastEditTime: 2021-06-25 14:18:17
FilePath: /home/zhihua/framework/deepspace/deepspace/agents/wp8/gan3d_simple_tpu.py
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

# from deepspace.graphs.losses.ssim import SSIM

from deepspace.agents.base import BasicAgent
from deepspace.graphs.models.autoencoder.ae3d_alpha import Residual3DAE
from deepspace.graphs.models.gan.dis3d_simple import Discriminator3D
from deepspace.graphs.layers.misc.wrapper import Wrapper
# from deepspace.datasets.wp8.npy_3d_break import NPYDataLoader
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
        self.generator = Residual3DAE(
            input_shape=config.deepspace.image_size,
            encoder_sizes=config.deepspace.gen_encoder_sizes,
            fc_sizes=config.deepspace.gen_fc_sizes,
            leak_value=config.deepspace.leak_value,
            temporal_strides=config.deepspace.gen_temporal_strides,
            color_channels=config.deepspace.image_channels,
        )
        self.discriminator = Discriminator3D(
            cube_len=config.deepspace.image_size,
            leak_value=config.deepspace.leak_value,
            bias=config.deepspace.bias,
        )
        # logger.info('========== device is =================== {device}, on {master}'.format(device=self.device, master='master' if self.master else 'worker'))
        # model go to tpu
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.generator.apply(xavier_weights)
        self.discriminator.apply(xavier_weights)

        # Create instance from the optimizer
        self.optimizer_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.deepspace.gen_lr if 'gen_lr' in config.deepspace else 1e-4,
        )
        self.optimizer_dis = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.deepspace.dis_lr if 'dis_lr' in config.deepspace else 1e-3,
        )
        # self.optimizer_dis = torch.optim.SGD(
        #     self.discriminator.parameters(),
        #     lr=config.deepspace.dis_lr,
        # )

        # define data_loader
        # load dataset with parallel data loader
        self.data_loader = Loader()
        if config.deepspace.mode == 'train':
            self.train_loader = parallel_loader.MpDeviceLoader(self.data_loader.train_loader, self.device)
            self.train_iterations = self.data_loader.train_iterations
            self.valid_loader = parallel_loader.MpDeviceLoader(self.data_loader.valid_loader, self.device)
            self.valid_iterations = self.data_loader.valid_iterations

            # Define Scheduler
            self.scheduler_gen = wrap_optimizer_with_scheduler(
                self.optimizer_gen,
                scheduler_type=config.deepspace.scheduler,
                scheduler_divisor=config.deepspace.scheduler_divisor,
                scheduler_divide_every_n_epochs=config.deepspace.scheduler_divide_every_n_epochs,
                num_steps_per_epoch=self.train_iterations,
                summary_writer=self.summary_writer
            )
            self.scheduler_dis = wrap_optimizer_with_scheduler(
                self.optimizer_dis,
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
        self.dis_loss = nn.BCELoss()
        # self.dis_loss = self.dis_loss.to(self.device)
        self.image_loss = nn.MSELoss()
        # self.image_loss = SSIM(
        #     data_range=1.0,
        #     size_average=True,
        #     channel=config.deepspace.image_channels,
        #     win_size=3,
        #     win_sigma=0.1,
        # )
        # self.image_loss = self.image_loss.to(self.device)
        # metric
        self.gen_best_metric = 100.0
        self.dis_best_metric = 100.0

        # positive and negative labels
        with torch.no_grad():
            self.real_labels = torch.FloatTensor(config.deepspace.train_batch).fill_(1).to(self.device)
            self.fake_labels = torch.FloatTensor(config.deepspace.train_batch).fill_(0).to(self.device)

        if self.master:
            # Tensorboard Writer
            self.summary_writer = SummaryWriter(log_dir=config.swap.summary_dir, comment='wp8 gan3d npy on tpu')
            # add model to tensorboard and print parameter to screen
            summary(self.generator, (1, config.deepspace.image_channels, *config.deepspace.image_size), device=self.device, depth=6)
            summary(self.discriminator, (1, config.deepspace.image_channels, *config.deepspace.image_size), device=self.device, depth=6)
            # add graph to tensorboard only if at epoch 0
            dummy_input = torch.randn(1, config.deepspace.image_channels, *config.deepspace.image_size).to(self.device)
            wrapper = Wrapper(self.generator, self.discriminator)
            self.summary_writer.add_graph(wrapper, (dummy_input), verbose=False)

        self.checkpoint = ['current_epoch', 'current_iteration', 'generator.state_dict',  'discriminator.state_dict', 'optimizer_gen.state_dict', 'optimizer_dis.state_dict']

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
            train_gen_loss, train_dis_loss = self.train_one_epoch()
            xla_model.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
            valid_gen_loss, valid_dis_loss = self.validate()
            self.scheduler_gen.step()
            self.scheduler_dis.step()
            is_best = valid_gen_loss < self.gen_best_metric or valid_dis_loss < self.dis_best_metric
            if valid_gen_loss < self.gen_best_metric:
                self.gen_best_metric = valid_gen_loss
            if valid_dis_loss < self.dis_best_metric:
                self.dis_best_metric = valid_dis_loss
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
            # desc="train on Epoch-{epoch}- on device- {device}/{ordinal}".format(epoch=self.current_epoch, device=self.device, ordinal=xla_model.get_ordinal())
            desc="train on Epoch-{epoch}/{max_epoch}-".format(epoch=self.current_epoch, max_epoch=config.deepspace.max_epoch)
        )
        # Set the model to be in training mode (for batchnorm)
        self.generator.train()
        self.discriminator.train()
        # Initialize average meters
        # Initialize average meters
        dis_epoch_loss = AverageMeter(device=self.device)
        dis_epoch_fake_loss = AverageMeter(device=self.device)
        dis_epoch_normal_loss = AverageMeter(device=self.device)
        gen_epoch_loss = AverageMeter(device=self.device)
        gen_epoch_dis_loss = AverageMeter(device=self.device)
        gen_epoch_image_loss = AverageMeter(device=self.device)
        # loop images
        for input_images, target_images in tqdm_batch:
            # input_images.requires_grad = True
            # target_images.requires_grad = True

            # 1.1 start the discriminator by training with real data---
            # Reset gradients
            self.optimizer_dis.zero_grad()
            out_labels = self.discriminator(target_images)
            dis_loss_normal = self.dis_loss(out_labels.squeeze(), self.real_labels[0:target_images.shape[0]])

            # 1.2 train the discriminator with fake data---
            fake_images_dis = self.generator(input_images)
            out_labels = self.discriminator(fake_images_dis.detach())
            dis_loss_fake = self.dis_loss(out_labels.squeeze(), self.fake_labels[0:input_images.shape[0]])
            # calculate total loss
            dis_loss = dis_loss_normal + dis_loss_fake
            dis_loss.backward()
            # Update weights with gradients: optimizer step and update loss to averager
            xla_model.optimizer_step(self.optimizer_dis)

            if self.current_epoch % config.deepspace.gen_epoch == 0:
                # train the generator now---
                self.optimizer_gen.zero_grad()
                fake_images_gen = self.generator(input_images)
                gen_image_loss = self.image_loss(fake_images_gen, target_images)
                out_labels = self.discriminator(fake_images_gen)
                # fake labels are real for generator cost
                gen_dis_loss = self.dis_loss(out_labels.squeeze(), self.real_labels[0:input_images.shape[0]])
                gen_loss = (1 - config.deepspace.loss_weight) * gen_dis_loss + config.deepspace.loss_weight * gen_image_loss
                gen_loss.backward()
                # Update weights with gradients: optimizer step and update loss to averager
                xla_model.optimizer_step(self.optimizer_gen)

            with torch.no_grad():
                dis_epoch_normal_loss.update(dis_loss_normal)
                dis_epoch_fake_loss.update(dis_loss_fake)
                dis_epoch_loss.update(dis_loss)
                if self.current_epoch % config.deepspace.gen_epoch == 0:
                    gen_epoch_dis_loss.update(gen_dis_loss)
                    gen_epoch_image_loss.update(gen_image_loss)
                    gen_epoch_loss.update(gen_loss)

            # update iteration counter
            self.current_iteration += 1
            tracker.add(config.deepspace.train_batch)

        # close dataloader
        tqdm_batch.close()

        # logging
        if self.master:
            if self.current_epoch % config.deepspace.gen_epoch == 0:
                self.summary_writer.add_scalar("epoch_training/gen_loss", gen_epoch_loss.val, self.current_iteration)
                self.summary_writer.add_scalar("epoch_training/gen_dis_loss", gen_epoch_dis_loss.val, self.current_iteration)
                self.summary_writer.add_scalar("epoch_training/gen_image_loss", gen_epoch_image_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_training/dis_loss", dis_epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_training/dis_normal_loss", dis_epoch_normal_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_training/dis_fake_loss", dis_epoch_fake_loss.val, self.current_iteration)
        # print info
        xla_model.master_print("Training Results at epoch-" + str(self.current_epoch)
                               + "\n" + "gen_loss: " + str(gen_epoch_loss.val)
                               + "\n" + "dis_loss: " + str(dis_epoch_loss.val)
                               + "\n" + "- gen_image_loss: " + str(gen_epoch_image_loss.val)
                               + "\n" + "- gen_dis_loss: " + str(gen_epoch_dis_loss.val)
                               + "\n" + "- dis_normal_loss: " + str(dis_epoch_normal_loss.val)
                               + "\n" + "- dis_fake_loss: " + str(dis_epoch_fake_loss.val))
        xla_model.add_step_closure(self.train_update, args=(self.device, self.current_iteration, gen_epoch_loss, tracker, self.current_epoch, self.summary_writer))
        return gen_epoch_loss.val, dis_epoch_loss.val

    def validate(self):
        # distribute tracker
        tracker = xla_model.RateTracker()
        tqdm_batch = tqdm(
            self.valid_loader,
            total=self.valid_iterations // xla_model.xrt_world_size(),
            # desc="validate at -{epoch}- on device- {device}/{ordinal}".format(epoch=self.current_epoch, device=self.device, ordinal=xla_model.get_ordinal())
            desc="validate at -{epoch}/{max_epoch}-".format(epoch=self.current_epoch, max_epoch=config.deepspace.max_epoch)
        )
        self.generator.eval()
        self.discriminator.eval()
        dis_epoch_loss = AverageMeter(device=self.device)
        dis_epoch_fake_loss = AverageMeter(device=self.device)
        dis_epoch_normal_loss = AverageMeter(device=self.device)
        gen_epoch_loss = AverageMeter(device=self.device)
        gen_epoch_dis_loss = AverageMeter(device=self.device)
        gen_epoch_image_loss = AverageMeter(device=self.device)
        recon_images_sample = []
        input_images_sample = []
        target_images_sample = []
        index = 0
        with torch.no_grad():
            for input_images, target_images in tqdm_batch:
                # test discriminator
                recon_images = self.generator(input_images)
                fake_labels = self.discriminator(recon_images)
                dis_fake_loss = self.dis_loss(fake_labels.squeeze(), self.fake_labels[0:input_images.shape[0]])
                normal_labels = self.discriminator(target_images)
                dis_normal_loss = self.dis_loss(normal_labels.squeeze(), self.real_labels[0:input_images.shape[0]])
                dis_loss = dis_fake_loss + dis_normal_loss
                dis_epoch_fake_loss.update(dis_fake_loss)
                dis_epoch_normal_loss.update(dis_normal_loss)
                dis_epoch_loss.update(dis_loss)
                # test generator
                gen_dis_loss = self.dis_loss(fake_labels.squeeze(), self.real_labels[0:input_images.shape[0]])
                gen_image_loss = self.image_loss(recon_images, target_images)
                gen_epoch_dis_loss.update(gen_dis_loss)
                gen_epoch_image_loss.update(gen_image_loss)
                gen_loss = (1 - config.deepspace.loss_weight) * gen_dis_loss + config.deepspace.loss_weight * gen_image_loss
                gen_epoch_loss.update(gen_loss)

                # save the reconstructed image
                if index >= config.deepspace.save_images[0] and index < config.deepspace.save_images[1]:
                    input_images = input_images.view(input_images.shape[0], input_images.shape[1], int(input_images.shape[2] * input_images.shape[3] / 8), -1)
                    recon_images = recon_images.view(recon_images.shape[0], recon_images.shape[1], int(recon_images.shape[2] * recon_images.shape[3] / 8), -1)
                    target_images = target_images.view(target_images.shape[0], target_images.shape[1], int(target_images.shape[2] * target_images.shape[3] / 8), -1)
                    input_images = input_images.squeeze().detach().cpu().numpy()
                    input_images_sample.append(np.expand_dims(input_images, axis=1))
                    target_images = target_images.squeeze().detach().cpu().numpy()
                    target_images_sample.append(np.expand_dims(target_images, axis=1))
                    recon_images = recon_images.squeeze().detach().cpu().numpy()
                    recon_images_sample.append(np.expand_dims(recon_images, axis=1))
                index = index + 1
                tracker.add(config.deepspace.validate_batch)

            tqdm_batch.close()

            if self.master:
                self.summary_writer.add_scalar("epoch_validate/gen_loss", gen_epoch_loss.val, self.current_iteration)
                self.summary_writer.add_scalar("epoch_validate/gen_dis_loss", gen_epoch_dis_loss.val, self.current_iteration)
                self.summary_writer.add_scalar("epoch_validate/gen_image_loss", gen_epoch_image_loss.val, self.current_iteration)
                self.summary_writer.add_scalar("epoch_validate/dis_loss", dis_epoch_loss.val, self.current_iteration)
                self.summary_writer.add_scalar("epoch_validate/dis_normal_loss", dis_epoch_normal_loss.val, self.current_iteration)
                self.summary_writer.add_scalar("epoch_validate/dis_fake_loss", dis_epoch_fake_loss.val, self.current_iteration)
                # save reconstruct image to tensorboard
                input_images_sample = np.concatenate(input_images_sample)
                target_images_sample = np.concatenate(target_images_sample)
                recon_images_sample = np.concatenate(recon_images_sample)
                recon_canvas = make_grid(recon_images_sample)
                self.summary_writer.add_image('recon_images', recon_canvas, self.current_epoch)
                input_canvas = make_grid(input_images_sample)
                self.summary_writer.add_image('input_images', input_canvas, self.current_epoch)
                target_canvas = make_grid(target_images_sample)
                self.summary_writer.add_image('target_images', target_canvas, self.current_epoch)
            # logging
            xla_model.master_print("validate Results at epoch-" + str(self.current_epoch)
                                   + "\n" + ' gen_loss: ' + str(gen_epoch_loss.val)
                                   + "\n" + ' dis_loss: ' + str(dis_epoch_loss.val)
                                   + "\n" + '- gen_image_loss: ' + str(gen_epoch_image_loss.val)
                                   + "\n" + '- gen_dis_loss: ' + str(gen_epoch_dis_loss.val)
                                   + "\n" + '- dis_normal_loss: ' + str(dis_epoch_normal_loss.val)
                                   + "\n" + '- dis_fake_loss: ' + str(dis_epoch_fake_loss.val))
            xla_model.add_step_closure(self.test_update, args=(self.device, self.current_iteration, gen_epoch_loss, self.current_epoch))
            return gen_epoch_loss.val, dis_epoch_loss.val

    def test(self):
        test_root = Path(config.deepspace.test_output_dir)
        tqdm_batch = tqdm(
            self.test_loader,
            total=self.test_iterations,
            # desc="test at -{epoch}- on device- {device}/{ordinal}".format(epoch=self.current_epoch, device=self.device, ordinal=xla_model.get_ordinal())
            desc="test at -{epoch}/{max_epoch}-".format(epoch=self.current_epoch, max_epoch=config.deepspace.max_epoch)
        )
        self.generator.eval()
        epoch_loss = AverageMeter(device=self.device)
        recon_images_sample = []
        input_images_sample = []
        image_paths = []
        with torch.no_grad():
            for images, paths in tqdm_batch:
                # images = images.to(self.device, dtype=torch.float32)
                # fake data---
                recon_images = self.generator(images)
                loss = self.image_loss(recon_images, images)
                epoch_loss.update(loss)

                # save the reconstructed image
                # recon_images = recon_images.squeeze().detach().cpu().numpy()
                recon_images_sample.append(recon_images)
                input_images_sample.append(images)
                image_paths.append(paths)

        tqdm_batch.close()

        image_paths = np.concatenate(np.array(image_paths))
        recon_images_sample = torch.cat(recon_images_sample).squeeze().detach().cpu().numpy()
        input_images_sample = torch.cat(input_images_sample).squeeze().detach().cpu().numpy()

        # recon_paths = [test_root / ('recon' + '_' + str(index) + '.' + config.deepspace.data_format) for index in range(recon_images_sample.shape[0])]
        # input_paths = [test_root / ('input' + '_' + str(index) + '.' + config.deepspace.data_format) for index in range(input_images_sample.shape[0])]
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
