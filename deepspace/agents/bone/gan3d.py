
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import lr_scheduler
from torch import nn
from tensorboardX.utils import make_grid
from tensorboardX import SummaryWriter
from torchsummary import summary

from deepspace.agents.base import BasicAgent
from deepspace.graphs.models.autoencoder.ae3d import Residual3DAE
from deepspace.graphs.models.gan.dis3d import Discriminator3D
from deepspace.graphs.layers.misc.wrapper import Wrapper
from deepspace.datasets.voxel.break_normal_npy import NPYDataLoader
from deepspace.graphs.weights_initializer import xavier_weights
from deepspace.utils.metrics import AverageMeter
from deepspace.utils.data import save_npy

from commontools.setup import config, logger


class BoneAgent(BasicAgent):
    def __init__(self):
        super().__init__()

        # define models
        self.generator = Residual3DAE(
            input_shape=config.deepspace.image_size,
            encoder_sizes=config.deepspace.gen_encoder_sizes,
            fc_sizes=config.deepspace.gen_fc_sizes,
            temporal_strides=config.deepspace.gen_temporal_strides,
            color_channels=config.deepspace.image_channels,
        )
        self.discriminator = Discriminator3D(
            input_shape=config.deepspace.image_size,
            encoder_sizes=config.deepspace.dis_encoder_sizes,
            fc_sizes=config.deepspace.dis_fc_sizes,
            temporal_strides=config.deepspace.dis_temporal_strides,
            color_channels=config.deepspace.image_channels,
            latent_activation=nn.Sigmoid()
        )
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # Create instance from the optimizer
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(),)
        self.optimizer_dis = torch.optim.Adam(self.discriminator.parameters(),)

        # Define Scheduler
        def lambda1(epoch): return pow(1 - epoch / config.deepspace.max_epoch, 0.9)
        self.scheduler_gen = lr_scheduler.LambdaLR(self.optimizer_gen, lr_lambda=lambda1)
        self.scheduler_dis = lr_scheduler.LambdaLR(self.optimizer_dis, lr_lambda=lambda1)

        self.generator.apply(xavier_weights)
        self.discriminator.apply(xavier_weights)

        # define data_loader
        self.data_loader = NPYDataLoader()

        # define loss
        self.dis_loss = nn.BCELoss()
        self.dis_loss = self.dis_loss.to(self.device)
        self.image_loss = nn.MSELoss()
        self.image_loss = self.image_loss.to(self.device)

        # metric
        self.gen_best_metric = 100.0
        self.dis_best_metric = 100.0

        self.checkpoint = ['current_epoch', 'current_iteration', 'generator.state_dict',  'discriminator.state_dict', 'optimizer_gen.state_dict', 'optimizer_dis.state_dict']

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(file_name=config.deepspace.checkpoint_file)

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=config.swap.summary_dir, comment='bone_gan3d_npy')
        # add model to tensorboard and print parameter to screen
        summary(self.generator, (config.deepspace.image_channels, *config.deepspace.image_size))
        summary(self.discriminator, (config.deepspace.image_channels, *config.deepspace.image_size))

        # add graph to tensorboard only if at epoch 0
        if self.current_epoch == 0:
            dummy_input = torch.randn(1, config.deepspace.image_channels, *config.deepspace.image_size).to(self.device)
            wrapper = Wrapper(self.generator, self.discriminator)
            self.summary_writer.add_graph(wrapper, (dummy_input), verbose=False)
            # self.summary_writer.add_graph(self.generator, (dummy_input), verbose=False)
            # self.summary_writer.add_graph(self.discriminator, (dummy_input), verbose=False)

        # positive and negative labels
        with torch.no_grad():
            self.real_labels = torch.FloatTensor(config.deepspace.train_batch).fill_(1).to(self.device)
            self.fake_labels = torch.FloatTensor(config.deepspace.train_batch).fill_(0).to(self.device)

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in tqdm(range(self.current_epoch, config.deepspace.max_epoch), desc="traing at -{}-, total epoches -{}-".format(self.current_epoch, config.deepspace.max_epoch)):
            self.current_epoch = epoch
            train_gen_loss, train_dis_loss = self.train_one_epoch()
            gen_loss, dis_loss = self.validate()
            self.scheduler_gen.step()
            self.scheduler_dis.step()
            is_best = gen_loss < self.gen_best_metric or dis_loss < self.dis_best_metric
            if gen_loss < self.gen_best_metric:
                self.gen_best_metric = gen_loss
            if dis_loss < self.dis_best_metric:
                self.dis_best_metric = dis_loss
            backup_checkpoint = False
            if self.current_epoch % config.deepspace.save_model_step == 0:
                backup_checkpoint = True
            self.save_checkpoint(config.deepspace.checkpoint_file,  is_best=is_best, backup_checkpoint=backup_checkpoint)

    def train_one_epoch(self):
        """
            One epoch of training
            :return:
            """
        # Initialize tqdm dataset
        tqdm_batch = tqdm(
            self.data_loader.train_loader,
            total=self.data_loader.train_iterations,
            desc="Epoch-{}-".format(self.current_epoch))
        # Set the model to be in training mode (for batchnorm)
        self.generator.train()
        self.discriminator.train()
        # Initialize average meters
        dis_epoch_loss = AverageMeter()
        dis_epoch_fake_loss = AverageMeter()
        dis_epoch_normal_loss = AverageMeter()
        gen_epoch_loss = AverageMeter()
        gen_epoch_dis_loss = AverageMeter()
        gen_epoch_image_loss = AverageMeter()
        # loop images
        for defect_images, normal_images in tqdm_batch:
            defect_images = defect_images.to(self.device, dtype=torch.float32)
            normal_images = normal_images.to(self.device, dtype=torch.float32)
            defect_images.requires_grad = True
            normal_images.requires_grad = True

            # 1.1 start the discriminator by training with real data---
            # Reset gradients
            self.optimizer_dis.zero_grad()
            out_labels = self.discriminator(normal_images)
            dis_loss_normal = self.dis_loss(out_labels.squeeze(), self.real_labels[0:normal_images.shape[0]])

            # 1.2 train the discriminator with fake data---
            fake_images = self.generator(defect_images)
            out_labels = self.discriminator(fake_images.detach())
            dis_loss_fake = self.dis_loss(out_labels.squeeze(), self.fake_labels[0:defect_images.shape[0]])
            # calculate total loss
            dis_loss = dis_loss_normal + dis_loss_fake
            dis_loss.backward()

            # Update weights with gradients: optimizer step and update loss to averager
            self.optimizer_dis.step()
            dis_epoch_normal_loss.update(dis_loss_normal.item())
            dis_epoch_fake_loss.update(dis_loss_fake.item())
            dis_epoch_loss.update(dis_loss.item())

            # train the generator now---
            self.optimizer_gen.zero_grad()
            out_labels = self.discriminator(fake_images)
            # fake labels are real for generator cost
            gen_dis_loss = self.dis_loss(out_labels.squeeze(), self.real_labels[0:defect_images.shape[0]])
            gen_image_loss = self.image_loss(fake_images, normal_images)
            gen_loss = (1 - config.deepspace.loss_weight) * gen_dis_loss + config.deepspace.loss_weight * gen_image_loss
            gen_loss.backward()
            # Update weights with gradients: optimizer step and update loss to averager
            self.optimizer_gen.step()
            gen_epoch_dis_loss.update(gen_dis_loss.item())
            gen_epoch_image_loss.update(gen_image_loss.item())
            gen_epoch_loss.update(gen_loss.item())

            # update iteration counter
            self.current_iteration += 1

        # close dataloader
        tqdm_batch.close()
        # logging
        self.summary_writer.add_scalar("epoch_training/gen_loss", gen_epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/gen_dis_loss", gen_epoch_dis_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/gen_image_loss", gen_epoch_image_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/dis_loss", dis_epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/dis_normal_loss", dis_epoch_normal_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/dis_fake_loss", dis_epoch_fake_loss.val, self.current_iteration)
        # print info
        logger.info("Training Results at epoch-" + str(self.current_epoch)
                    + "\n" + "gen_loss: " + str(gen_epoch_loss.val)
                    + "\n" + "dis_loss: " + str(dis_epoch_loss.val)
                    + "\n" + "- gen_image_loss: " + str(gen_epoch_image_loss.val)
                    + "\n" + "- gen_dis_loss: " + str(gen_epoch_dis_loss.val)
                    + "\n" + "- dis_normal_loss: " + str(dis_epoch_normal_loss.val)
                    + "\n" + "- dis_fake_loss: " + str(dis_epoch_fake_loss.val))

        return gen_epoch_loss.val, dis_epoch_loss.val

    def validate(self):
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations, desc="validate at -{}-".format(self.current_epoch))
        self.generator.eval()
        self.discriminator.eval()
        dis_epoch_loss = AverageMeter()
        dis_epoch_fake_loss = AverageMeter()
        dis_epoch_normal_loss = AverageMeter()
        gen_epoch_loss = AverageMeter()
        gen_epoch_dis_loss = AverageMeter()
        gen_epoch_image_loss = AverageMeter()
        fake_images_sample = []
        normal_images_sample = []
        index = 0
        with torch.no_grad():
            for defect_images, normal_images in tqdm_batch:
                defect_images = defect_images.to(self.device, dtype=torch.float32)
                normal_images = normal_images.to(self.device, dtype=torch.float32)
                # test discriminator
                fake_images = self.generator(defect_images)
                fake_labels = self.discriminator(fake_images)
                dis_fake_loss = self.dis_loss(fake_labels.squeeze(), self.fake_labels[0:defect_images.shape[0]])
                normal_labels = self.discriminator(normal_images)
                dis_normal_loss = self.dis_loss(normal_labels.squeeze(), self.real_labels[0:defect_images.shape[0]])
                dis_loss = dis_fake_loss + dis_normal_loss
                dis_epoch_fake_loss.update(dis_fake_loss.item())
                dis_epoch_normal_loss.update(dis_normal_loss.item())
                dis_epoch_loss.update(dis_loss.item())
                # test generator
                gen_dis_loss = self.dis_loss(fake_labels.squeeze(), self.real_labels[0:defect_images.shape[0]])
                gen_image_loss = self.image_loss(fake_images, normal_images)
                gen_epoch_dis_loss.update(gen_dis_loss.item())
                gen_epoch_image_loss.update(gen_image_loss.item())
                gen_loss = (1 - config.deepspace.loss_weight) * gen_dis_loss + config.deepspace.loss_weight * gen_image_loss
                gen_epoch_loss.update(gen_loss.item())

                # save the reconstructed image
                if index > config.deepspace.save_images[0] and index < config.deepspace.save_images[1]:
                    fake_images = fake_images.view(fake_images.shape[0], fake_images.shape[1], int(fake_images.shape[2] * fake_images.shape[3] / 8), -1)
                    normal_images = normal_images.view(normal_images.shape[0], normal_images.shape[1], int(normal_images.shape[2] * normal_images.shape[3] / 8), -1)
                    fake_images = fake_images.squeeze().detach().cpu().numpy()
                    fake_images_sample.append(np.expand_dims(fake_images, axis=1))
                    normal_images = normal_images.squeeze().detach().cpu().numpy()
                    normal_images_sample.append(np.expand_dims(normal_images, axis=1))
                index = index + 1

            tqdm_batch.close()

            # logging
            self.summary_writer.add_scalar("epoch_validate/gen_loss", gen_epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_validate/gen_dis_loss", gen_epoch_dis_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_validate/gen_image_loss", gen_epoch_image_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_validate/dis_loss", dis_epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_validate/dis_normal_loss", dis_epoch_normal_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_validate/dis_fake_loss", dis_epoch_fake_loss.val, self.current_iteration)
            # save reconstruct image to tensorboard
            fake_images_sample = np.concatenate(fake_images_sample)
            normal_images_sample = np.concatenate(normal_images_sample)
            recon_canvas = make_grid(fake_images_sample)
            self.summary_writer.add_image('fake_images', recon_canvas, self.current_epoch)
            input_canvas = make_grid(normal_images_sample)
            self.summary_writer.add_image('normal_images', input_canvas, self.current_epoch)
            # print info
            logger.info("validate Results at epoch-" + str(self.current_epoch)
                        + "\n" + ' gen_loss: ' + str(gen_epoch_loss.val)
                        + "\n" + ' dis_loss: ' + str(dis_epoch_loss.val)
                        + "\n" + '- gen_image_loss: ' + str(gen_epoch_image_loss.val)
                        + "\n" + '- gen_dis_loss: ' + str(gen_epoch_dis_loss.val)
                        + "\n" + '- dis_normal_loss: ' + str(dis_epoch_normal_loss.val)
                        + "\n" + '- dis_fake_loss: ' + str(dis_epoch_fake_loss.val))

            return gen_epoch_loss.val, dis_epoch_loss.val

    def test(self):
        size = config.deepspace.image_size
        recon_data = np.zeros_like(self.data_loader.test_data)
        recon_data = np.squeeze(recon_data)
        tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations, desc="test at -{}-".format(self.current_epoch))
        self.generator.eval()
        self.discriminator.eval()
        epoch_loss = AverageMeter()
        gen_epoch_loss = AverageMeter()
        gen_epoch_dis_loss = AverageMeter()
        gen_epoch_image_loss = AverageMeter()
        dis_epoch_loss = AverageMeter()

        with torch.no_grad():
            for images, coordinate in tqdm_batch:
                images = images.to(self.device, dtype=torch.float32)
                # fake data---
                recon_images = self.generator(images)
                fake_labels = self.discriminator(recon_images)
                # train the model now---
                gen_image_loss = self.image_loss(recon_images, images)
                gen_dis_loss = self.dis_loss(fake_labels.squeeze(), self.real_labels[0:images.shape[0]])
                dis_loss = self.dis_loss(fake_labels.squeeze(), self.fake_labels[0:images.shape[0]])
                gen_loss = (1 - config.deepspace.loss_weight) * gen_dis_loss + config.deepspace.loss_weight * gen_image_loss

                gen_epoch_loss.update(gen_loss.item())
                gen_epoch_image_loss.update(gen_image_loss.item())
                gen_epoch_dis_loss.update(gen_dis_loss.item())
                dis_epoch_loss.update(dis_loss.item())
                # save the reconstructed image
                recon_images = recon_images.squeeze().detach().cpu().numpy()
                for image, cord in zip(recon_images, coordinate):
                    recon_data[cord[0]:cord[0]+size[0], cord[1]:cord[1]+size[1], cord[2]:cord[2]+size[2]] = image

            # data.shape should be (N, h, w, d)
            np.save(Path(config.deepspace.test_output_dir) / 'output.npy', recon_data)
            # logging
            logger.info("test Results at epoch-" + str(self.current_epoch)
                        + "\n" + ' gen_loss: ' + str(gen_epoch_loss.val)
                        + "\n" + ' dis_loss: ' + str(dis_epoch_loss.val)
                        + "\n" + '- gen_image_loss: ' + str(gen_epoch_image_loss.val)
                        + "\n" + '- gen_dis_loss: ' + str(gen_epoch_dis_loss.val))

            logger.info("test Results at epoch-" + str(self.current_epoch) + " | " + ' epoch_loss: ' + str(epoch_loss.val) + " | ")
            tqdm_batch.close()
            return epoch_loss.val
