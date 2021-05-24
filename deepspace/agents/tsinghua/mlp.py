
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
from deepspace.graphs.models.mlp.tsinghua import MLP
from deepspace.datasets.tsinghua.npy import NPYDataLoader
from deepspace.graphs.losses.mse import NeutrinoLoss
from deepspace.graphs.weights_initializer import xavier_weights
from deepspace.utils.metrics import AverageMeter
from deepspace.utils.data import normalization

from commontools.setup import config, logger


class NeutrinoAgent(BasicAgent):
    def __init__(self):
        super().__init__()

        # define models
        self.model = MLP(
            input_shape=config.deepspace.shape,
            wave_mlp_sizes=config.deepspace.wave_mlp_sizes,
            det_mlp_sizes=config.deepspace.det_mlp_sizes,
            activation=nn.Sigmoid(),
        )
        self.model = self.model.to(self.device)

        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.deepspace.learning_rate, )

        # Define Scheduler
        def lambda1(epoch): return pow(1 - epoch / config.deepspace.max_epoch, config.deepspace.lr_decay)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        self.model.apply(xavier_weights)

        # define data_loader
        self.data_loader = NPYDataLoader()

        # define loss
        self.loss = NeutrinoLoss()
        self.loss = self.loss.to(self.device)

        # metric
        self.best_metric = 100.0

        self.checkpoint = ['current_epoch', 'current_iteration', 'model.state_dict',  'optimizer.state_dict']

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(file_name=config.deepspace.checkpoint_file)

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=config.swap.summary_dir, comment='tsinghua-neutrino-mlp')
        # add model to tensorboard and print parameter to screen
        summary(self.model, input_size=[(1, 3000), (1, 3000, 1000)], dtypes=[torch.int64, torch.float], device=self.device)

        # add graph to tensorboard only if at epoch 0
        if self.current_epoch == 0:
            dummy_input = torch.randn(1, 300, 1000).to(self.device)
            index = torch.from_numpy(np.random.randint(0, 43212, [1, 300])).to(self.device)
            self.summary_writer.add_graph(self.model, [index, dummy_input], verbose=False)

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in tqdm(range(self.current_epoch, config.deepspace.max_epoch), desc="total epoches -{}-".format(config.deepspace.max_epoch)):
            self.current_epoch = epoch
            train_loss = self.train_one_epoch()
            loss = self.validate()
            self.scheduler.step()
            is_best = loss < self.best_metric
            if loss < self.best_metric:
                self.best_metric = loss
            backup_checkpoint = False
            if self.current_epoch % config.deepspace.save_model_step == 0:
                backup_checkpoint = True
            if self.current_epoch % config.deepspace.save_checkpoint_step == 0:
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
        self.model.train()
        # Initialize average meters
        epoch_loss = AverageMeter()
        # loop datas
        for index, data, label in tqdm_batch:
            # data to device, autograd
            index = index.to(self.device, dtype=torch.int64)
            data = data.to(self.device, dtype=torch.float32)
            label = label.to(self.device, dtype=torch.float64)
            # index.requires_grad = True
            data.requires_grad = True
            label.requires_grad = True

            # train the model now---
            self.optimizer.zero_grad()
            pred_label = self.model(index, data)
            loss = self.loss(pred_label.squeeze(), label)
            loss.backward()
            # Update weights with gradients: optimizer step and update loss to averager
            self.optimizer.step()
            epoch_loss.update(loss.item())
            # update iteration counter
            self.current_iteration += 1

        # close dataloader
        tqdm_batch.close()
        # logging
        self.summary_writer.add_scalar("epoch_training/loss", epoch_loss.val, self.current_epoch)
        # print info
        logger.info("Training Results at epoch-" + str(self.current_epoch) + "loss: " + str(epoch_loss.val))
        return epoch_loss.val

    def validate(self):
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations, desc="validate at -{}-".format(self.current_epoch))
        self.model.eval()
        epoch_loss = AverageMeter()
        with torch.no_grad():
            for index, data, label in tqdm_batch:
                # data to device, autograd
                index = index.to(self.device, dtype=torch.int64)
                data = data.to(self.device, dtype=torch.float32)
                label = label.to(self.device, dtype=torch.float64)

                # test model
                pred_label = self.model(index, data)
                loss = self.loss(pred_label.squeeze(), label)
                epoch_loss.update(loss.item())

                # save the waveform image
                if self.current_epoch % config.deepspace.save_image_step == 0:
                    data = data.squeeze().detach().cpu().numpy()

            tqdm_batch.close()

            # logging
            self.summary_writer.add_scalar("epoch_validate/loss", epoch_loss.val, self.current_epoch)

            # print info
            logger.info("validate Results at epoch-" + str(self.current_epoch) + ' loss: ' + str(epoch_loss.val))
            return epoch_loss.val

    def test(self):
        recon_data = []
        tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations, desc="test at -{}-".format(self.current_epoch))
        self.model.eval()
        self.discriminator.eval()
        epoch_loss = AverageMeter()
        epoch_loss = AverageMeter()
        gen_epoch_dis_loss = AverageMeter()
        gen_epoch_image_loss = AverageMeter()
        dis_epoch_loss = AverageMeter()

        with torch.no_grad():
            for images, coordinate in tqdm_batch:
                images = images.to(self.device, dtype=torch.float32)
                # fake data---
                recon_images = self.model(images)
                fake_labels = self.discriminator(recon_images)
                # train the model now---
                gen_image_loss = self.loss(recon_images, images)
                gen_dis_loss = self.dis_loss(fake_labels.squeeze(), self.real_labels[0:images.shape[0]])
                dis_loss = self.dis_loss(fake_labels.squeeze(), self.fake_labels[0:images.shape[0]])
                loss = (1 - config.deepspace.loss_weight) * gen_dis_loss + config.deepspace.loss_weight * gen_image_loss

                epoch_loss.update(loss.item())
                gen_epoch_image_loss.update(gen_image_loss.item())
                gen_epoch_dis_loss.update(gen_dis_loss.item())
                dis_epoch_loss.update(dis_loss.item())
                # save the reconstructed image
                recon_images = recon_images.squeeze().detach().cpu().numpy()
                recon_data.append(recon_images)

            # data.shape should be (N, h, w, d)
            recon_data = np.stack(recon_data)
            np.save(Path(config.deepspace.test_output_dir) / 'output.npy', recon_data)
            # logging
            logger.info("test Results at epoch-" + str(self.current_epoch)
                        + "\n" + ' loss: ' + str(epoch_loss.val)
                        + "\n" + ' dis_loss: ' + str(dis_epoch_loss.val)
                        + "\n" + '- gen_image_loss: ' + str(gen_epoch_image_loss.val)
                        + "\n" + '- gen_dis_loss: ' + str(gen_epoch_dis_loss.val))

            logger.info("test Results at epoch-" + str(self.current_epoch) + " | " + ' epoch_loss: ' + str(epoch_loss.val) + " | ")
            tqdm_batch.close()
            return epoch_loss.val
