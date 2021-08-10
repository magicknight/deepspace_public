
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import lr_scheduler
from torch import nn
from tensorboardX.utils import make_grid
from tensorboardX import SummaryWriter
from torchinfo import summary

from deepspace.agents.base import BasicAgent
from deepspace.graphs.models.autoencoder.ae2dn import ResidualAE
from deepspace.graphs.models.gan.dis2dn import Discriminator
from deepspace.graphs.layers.misc.wrapper import Wrapper
from deepspace.datasets.voxel.npy_2d_tpu import Loader
from deepspace.graphs.weights_initializer import xavier_weights
from deepspace.utils.metrics import AverageMeter
from deepspace.utils.data import save_npy
# from deepspace.augmentation.diff_aug_simple import DiffAugment
from deepspace.graphs.scheduler.scheduler import wrap_optimizer_with_scheduler
from deepspace.graphs.losses.ssim import SSIM_Loss, SSIM


from commontools.setup import config, logger
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
        self.generator = ResidualAE(
            input_shape=config.deepspace.image_size,
            encoder_sizes=config.deepspace.gen_encoder_sizes,
            fc_sizes=config.deepspace.gen_fc_sizes,
            temporal_strides=config.deepspace.gen_temporal_strides,
            color_channels=config.deepspace.image_channels,
        )
        self.discriminator = Discriminator(
            input_shape=config.deepspace.image_size,
            encoder_sizes=config.deepspace.dis_encoder_sizes,
            fc_sizes=config.deepspace.dis_fc_sizes,
            temporal_strides=config.deepspace.dis_temporal_strides,
            color_channels=config.deepspace.image_channels,
            latent_activation=nn.Sigmoid()
        )
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.generator.apply(xavier_weights)
        self.discriminator.apply(xavier_weights)

        # Create instance from the optimizer
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=config.deepspace.gen_lr)
        self.optimizer_dis = torch.optim.Adam(self.discriminator.parameters(), lr=config.deepspace.dis_lr)

        # define data_loader
        # load dataset with parallel data loader
        self.data_loader = Loader(shared_array=self.shared_array)
        if config.deepspace.mode == 'train':
            self.train_loader = parallel_loader.MpDeviceLoader(self.data_loader.train_loader, self.device)
            self.train_iterations = self.data_loader.train_iterations
            self.valid_loader = parallel_loader.MpDeviceLoader(self.data_loader.valid_loader, self.device)
            self.valid_iterations = self.data_loader.valid_iterations
        elif config.deepspace.mode == 'test':
            if config.common.distribute:
                self.test_loader = parallel_loader.MpDeviceLoader(self.data_loader.test_loader, self.device)
                self.test_iterations = self.data_loader.test_iterations // xla_model.xrt_world_size()
            else:
                # if not on distribution.
                self.test_loader = self.data_loader.test_loader
                self.test_iterations = self.data_loader.test_iterations

        # Define Scheduler
        # def lambda1(epoch): return pow(1 - epoch / config.deepspace.max_epoch, config.deepspace.lr_decay)
        # self.scheduler_gen = lr_scheduler.ExponentialLR(self.optimizer_gen, gamma=config.deepspace.lr_decay)
        # self.scheduler_dis = lr_scheduler.ExponentialLR(self.optimizer_dis, gamma=config.deepspace.lr_decay)
        self.scheduler_gen = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_gen, T_0=config.deepspace.T_0, T_mult=config.deepspace.T_mult, )
        self.scheduler_dis = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_dis, T_0=config.deepspace.T_0, T_mult=config.deepspace.T_mult, )
        # self.scheduler_gen = wrap_optimizer_with_scheduler(
        #     self.optimizer_gen,
        #     scheduler_type=config.deepspace.scheduler,
        #     scheduler_divisor=config.deepspace.scheduler_divisor,
        #     scheduler_divide_every_n_epochs=config.deepspace.scheduler_divide_every_n_epochs,
        #     num_steps_per_epoch=self.train_iterations,
        #     summary_writer=self.summary_writer
        # )
        # self.scheduler_dis = wrap_optimizer_with_scheduler(
        #     self.optimizer_dis,
        #     scheduler_type=config.deepspace.scheduler,
        #     scheduler_divisor=config.deepspace.scheduler_divisor,
        #     scheduler_divide_every_n_epochs=config.deepspace.scheduler_divide_every_n_epochs,
        #     num_steps_per_epoch=self.train_iterations,
        #     summary_writer=self.summary_writer
        # )

        # define loss
        self.dis_loss = nn.BCELoss()
        self.image_loss = nn.MSELoss()
        # self.image_loss = nn.L1Loss()
        # self.image_loss = SSIM_Loss()

        # metric
        self.gen_best_metric = 100.0
        self.dis_best_metric = 100.0

        # positive and negative labels
        with torch.no_grad():
            self.real_labels = torch.FloatTensor(config.deepspace.train_batch, 1).fill_(1.0).to(self.device).requires_grad_(True)
            self.fake_labels = torch.FloatTensor(config.deepspace.train_batch, 1).fill_(0.0).to(self.device).requires_grad_(True)

        if self.master:
            # Tensorboard Writer
            self.summary_writer = SummaryWriter(log_dir=config.swap.summary_dir, comment='bone gan2d npy on tpu')
            # add model to tensorboard and print parameter to screen
            summary(self.generator, (1, config.deepspace.image_channels, *config.deepspace.image_size), device=self.device, depth=6)
            summary(self.discriminator, (1, config.deepspace.image_channels, *config.deepspace.image_size), device=self.device, depth=6)
            # add graph to tensorboard only if at epoch 0
            if self.current_epoch == 0:
                dummy_input = torch.randn(1, config.deepspace.image_channels, *config.deepspace.image_size).to(self.device)
                wrapper = Wrapper(self.generator, self.discriminator)
                self.summary_writer.add_graph(wrapper, (dummy_input), verbose=False)

        self.checkpoint = [
            'current_epoch', 'current_iteration', 'generator.state_dict',  'discriminator.state_dict',
            'optimizer_gen.state_dict', 'optimizer_dis.state_dict', 'scheduler_gen.state_dict', 'scheduler_dis.state_dict'
        ]

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(file_name=config.deepspace.checkpoint_file)

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, config.deepspace.max_epoch):
            xla_model.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
            self.current_epoch = epoch
            train_gen_loss, train_dis_loss = self.train_one_epoch()
            xla_model.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
            if 'validate_step' in config.deepspace and self.current_epoch % config.deepspace.validate_step != 0:
                continue
            valid_gen_loss, valid_dis_loss = self.validate()
            self.scheduler_gen.step()
            self.scheduler_dis.step()
            is_best = valid_gen_loss < self.gen_best_metric or valid_dis_loss < self.dis_best_metric
            if valid_gen_loss < self.gen_best_metric:
                self.gen_best_metric = valid_gen_loss
            if valid_dis_loss < self.dis_best_metric:
                self.dis_best_metric = valid_dis_loss
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
        self.generator.train()
        self.discriminator.train()
        # Initialize average meters
        dis_epoch_loss = AverageMeter(device=self.device)
        dis_epoch_fake_loss = AverageMeter(device=self.device)
        dis_epoch_normal_loss = AverageMeter(device=self.device)
        gen_epoch_loss = AverageMeter(device=self.device)
        gen_epoch_dis_loss = AverageMeter(device=self.device)
        gen_epoch_image_loss = AverageMeter(device=self.device)
        # loop images
        for input_images, target_images in tqdm_batch:
            input_images.requires_grad = True
            target_images.requires_grad = True
            # xla_model.master_print(input_images.device, input_images.dtype, input_images.requires_grad, input_images.shape)
            # xla_model.master_print(target_images.device, target_images.dtype, target_images.requires_grad, target_images.shape)
            # 1.1 start the discriminator by training with real data---
            # Reset gradients
            self.optimizer_dis.zero_grad()
            # out_labels_normal = self.discriminator(DiffAugment(target_images, policy=config.deepspace.policy))
            out_labels_normal = self.discriminator(target_images)
            dis_loss_normal = self.dis_loss(out_labels_normal, self.real_labels[0:target_images.shape[0]])

            # 1.2 train the discriminator with fake data---
            fake_images = self.generator(input_images)
            # out_labels_fake = self.discriminator(DiffAugment(fake_images, policy=config.deepspace.policy))
            out_labels_fake = self.discriminator(fake_images)
            dis_loss_fake = self.dis_loss(out_labels_fake, self.fake_labels[0:input_images.shape[0]])

            # calculate total loss
            dis_loss = dis_loss_normal + dis_loss_fake
            dis_loss.backward(retain_graph=True)

            # Update weights with gradients: optimizer step and update loss to averager
            # self.optimizer_dis.step()
            xla_model.optimizer_step(self.optimizer_dis, barrier=True)
            # xla_model.reduce_gradients(self.optimizer_dis)
            # # clip(optimizer)
            # self.optimizer_dis.step()

            # train the generator now---
            self.optimizer_gen.zero_grad()
            gen_image_loss = self.image_loss(fake_images, target_images)
            # out_labels = self.discriminator(DiffAugment(fake_images, policy=config.deepspace.policy))
            out_labels = self.discriminator(fake_images)
            # fake labels are real for generator cost
            gen_dis_loss = self.dis_loss(out_labels, self.real_labels[0:input_images.shape[0]])
            gen_loss = (1 - config.deepspace.loss_weight) * gen_dis_loss + config.deepspace.loss_weight * gen_image_loss
            gen_loss.backward()
            # Update weights with gradients: optimizer step and update loss to averager
            # self.optimizer_gen.step()
            xla_model.optimizer_step(self.optimizer_gen, barrier=True)
            # xla_model.reduce_gradients(self.optimizer_gen)
            # # clip(optimizer)
            # self.optimizer_gen.step()

            dis_epoch_normal_loss.update(dis_loss_normal)
            dis_epoch_fake_loss.update(dis_loss_fake)
            dis_epoch_loss.update(dis_loss)
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
            self.summary_writer.add_scalar("epoch_training/gen_loss", gen_epoch_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_training/gen_dis_loss", gen_epoch_dis_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_training/gen_image_loss", gen_epoch_image_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_training/dis_loss", dis_epoch_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_training/dis_normal_loss", dis_epoch_normal_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_training/dis_fake_loss", dis_epoch_fake_loss.val, self.current_epoch)
        # print info
        xla_model.master_print("Training Results at epoch-" + str(self.current_epoch)
                               + "\n" + "--------------------------------"
                               + "\n" + "gen_loss: " + str(gen_epoch_loss.val)
                               + "\n" + "- gen_image_loss: " + str(gen_epoch_image_loss.val)
                               + "\n" + "- gen_dis_loss: " + str(gen_epoch_dis_loss.val)
                               + "\n" + "--------------------------------"
                               + "\n" + "dis_loss: " + str(dis_epoch_loss.val)
                               + "\n" + "- dis_normal_loss: " + str(dis_epoch_normal_loss.val)
                               + "\n" + "- dis_fake_loss: " + str(dis_epoch_fake_loss.val))
        xla_model.add_step_closure(self.train_update, args=(self.device, self.current_iteration, gen_epoch_loss, tracker, self.current_epoch, self.summary_writer))
        if torch.isnan(gen_epoch_loss.avg) or torch.isnan(dis_epoch_loss.avg):
            raise ValueError("NaN detected!")
        return gen_epoch_loss.val, dis_epoch_loss.val

    @torch.no_grad()
    def validate(self):
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
        target_images_sample = []
        input_images_sample = []
        index = 0
        # for each batch
        for input_images, target_images in tqdm_batch:
            # test discriminator
            recon_images = self.generator(input_images)
            fake_labels = self.discriminator(recon_images)
            dis_fake_loss = self.dis_loss(fake_labels, self.fake_labels[0:input_images.shape[0]])
            normal_labels = self.discriminator(target_images)
            dis_normal_loss = self.dis_loss(normal_labels, self.real_labels[0:input_images.shape[0]])
            dis_loss = dis_fake_loss + dis_normal_loss
            dis_epoch_fake_loss.update(dis_fake_loss)
            dis_epoch_normal_loss.update(dis_normal_loss)
            dis_epoch_loss.update(dis_loss)
            # test generator
            gen_dis_loss = self.dis_loss(fake_labels, self.real_labels[0:input_images.shape[0]])
            gen_image_loss = self.image_loss(recon_images, target_images)
            gen_epoch_dis_loss.update(gen_dis_loss)
            gen_epoch_image_loss.update(gen_image_loss)
            gen_loss = (1 - config.deepspace.loss_weight) * gen_dis_loss + config.deepspace.loss_weight * gen_image_loss
            gen_epoch_loss.update(gen_loss)

            tracker.add(config.deepspace.validate_batch)

            # save the reconstructed image
            if self.master and self.current_epoch % config.deepspace.save_image_step == 0:
                if index > config.deepspace.save_images[0] and index < config.deepspace.save_images[1]:
                    recon_images_sample.append(recon_images)
                    target_images_sample.append(target_images)
                    input_images_sample.append(input_images)
            index = index + 1

        tqdm_batch.close()

        # logging
        if self.master:
            self.summary_writer.add_scalar("epoch_validate/gen_loss", gen_epoch_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_validate/gen_dis_loss", gen_epoch_dis_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_validate/gen_image_loss", gen_epoch_image_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_validate/dis_loss", dis_epoch_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_validate/dis_normal_loss", dis_epoch_normal_loss.val, self.current_epoch)
            self.summary_writer.add_scalar("epoch_validate/dis_fake_loss", dis_epoch_fake_loss.val, self.current_epoch)
            # save reconstruct image to tensorboard
            if self.current_epoch % config.deepspace.save_image_step == 0:
                recon_images_sample = torch.cat(recon_images_sample).detach().cpu().numpy()
                target_images_sample = torch.cat(target_images_sample).detach().cpu().numpy()
                input_images_sample = torch.cat(input_images_sample).detach().cpu().numpy()
                recon_canvas = make_grid(recon_images_sample)
                self.summary_writer.add_image('recon_images', recon_canvas, self.current_epoch)
                input_canvas = make_grid(target_images_sample)
                self.summary_writer.add_image('target_images', input_canvas, self.current_epoch)
                defect_canvas = make_grid(input_images_sample)
                self.summary_writer.add_image('input_images', defect_canvas, self.current_epoch)
            # print info
        xla_model.master_print("validate Results at epoch-" + str(self.current_epoch)
                               + "\n" + "--------------------------------"
                               + "\n" + ' gen_loss: ' + str(gen_epoch_loss.val)
                               + "\n" + '- gen_image_loss: ' + str(gen_epoch_image_loss.val)
                               + "\n" + '- gen_dis_loss: ' + str(gen_epoch_dis_loss.val)
                               + "\n" + "--------------------------------"
                               + "\n" + ' dis_loss: ' + str(dis_epoch_loss.val)
                               + "\n" + '- dis_normal_loss: ' + str(dis_epoch_normal_loss.val)
                               + "\n" + '- dis_fake_loss: ' + str(dis_epoch_fake_loss.val))
        xla_model.add_step_closure(self.test_update, args=(self.device, self.current_iteration, gen_epoch_loss, self.current_epoch))
        return gen_epoch_loss.val, dis_epoch_loss.val

    @torch.no_grad()
    def test(self):
        input_data = []
        recon_data = []
        normal_data = []
        tqdm_batch = tqdm(self.test_loader, total=self.data_loader.test_iterations, desc="test at -{}-".format(self.current_epoch))
        self.generator.eval()
        self.discriminator.eval()
        gen_epoch_image_loss = AverageMeter(device=self.device)

        with torch.no_grad():
            for input_images, normal_images in tqdm_batch:
                input_images = input_images.to(self.device, dtype=torch.float32)
                # fake data---
                normal_images = normal_images.to(self.device, dtype=torch.float32)
                # test discriminator
                recon_images = self.generator(input_images)

                # test generator
                gen_image_loss = self.image_loss(recon_images, normal_images)
                gen_epoch_image_loss.update(gen_image_loss)

                # save the  images
                input_images = input_images.squeeze().detach().cpu().numpy()
                recon_images = recon_images.squeeze().detach().cpu().numpy()
                normal_images = normal_images.squeeze().detach().cpu().numpy()
                input_data += [*input_images]
                recon_data += [*recon_images]
                normal_data += [*normal_images]

            # data.shape should be (N, w, d)
            input_data = np.stack(input_data)
            recon_data = np.stack(recon_data)
            normal_data = np.stack(normal_data)
            np.save(Path(config.deepspace.test_output_dir) / 'input.npy', input_data)
            np.save(Path(config.deepspace.test_output_dir) / 'recon.npy', recon_data)
            np.save(Path(config.deepspace.test_output_dir) / 'normal.npy', normal_data)
            # logging
            logger.info("test Results at epoch-" + str(self.current_epoch) + "\n" + '- gen_image_loss: ' + str(gen_epoch_image_loss.val))
            tqdm_batch.close()
            return gen_epoch_image_loss.val

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
