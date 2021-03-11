"""
Defect detection agent
"""
import numpy as np
from numpy.testing._private.utils import requires_memory

from tqdm import tqdm
import shutil
import random
from pathlib import Path

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter

from deepspace.agents.base import BaseAgent

from deepspace.graphs.models.autoencoder import AutoEncoder
from deepspace.datasets.defect import DefectDataLoader
from deepspace.graphs.losses.ssim import SSIM_Loss, ssim, MS_SSIM_Loss, ms_ssim

from deepspace.utils.metrics import AverageMeter, AverageMeterList
from deepspace.utils.misc import print_cuda_statistics
from deepspace.utils.data import to_uint8, save_images, make_heatmaps
from deepspace.utils.dirs import create_dirs
from deepspace.config.config import config, logger

cudnn.benchmark = True


class DefectAgent(BaseAgent):

    def __init__(self):
        super().__init__(config)

        # set cuda flag
        config.settings.cuda = config.settings.device == 'gpu'
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not config.settings.cuda:
            logger.warn("You have a CUDA device, so you should probably enable CUDA!")

        self.cuda = self.is_cuda & config.settings.cuda

        # set the manual seed for torch
        if self.cuda:
            torch.cuda.manual_seed_all(config.settings.seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(config.settings.gpu_device)
            logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(config.settings.seed)
            logger.info("Program will run on *****CPU*****\n")

        # define models
        self.model = AutoEncoder(C=config.settings.model_c, M=config.settings.model_m, in_chan=config.settings.image_channels, out_chan=config.settings.image_channels)
        self.model = self.model.to(self.device)

        # define data_loader
        self.data_loader = DefectDataLoader()

        # define loss
        self.loss = SSIM_Loss(channel=config.settings.image_channels, data_range=1.0, size_average=True) if config.settings.loss == 'ssim' else MS_SSIM_Loss(channel=config.settings.image_channels, data_range=1.0, size_average=True)
        self.loss = self.loss.to(self.device)

        # define metrics
        self.metrics = ssim if config.settings.loss == 'ssim' else ms_ssim

        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config.settings.learning_rate,
                                          #   betas=config.settings.betas,
                                          #   eps=config.settings.eps,
                                          weight_decay=config.settings.weight_decay)

        # Define Scheduler
        def lambda1(epoch): return pow(1 - epoch / config.settings.max_epoch, 0.9)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(config.settings.checkpoint_file)
        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=config.settings.summary_dir, comment='FCN8s')

        # # scheduler for the optimizer
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                                             'min', patience=config.learning_rate_patience,
        #                                                             min_lr=1e-10, verbose=True)

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        file_name = config.settings.checkpoint_dir + file_name
        try:
            logger.info("Loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                        .format(config.settings.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            logger.info("No checkpoint exists from '{}'. Skipping...".format(config.settings.checkpoint_dir))
            logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, config.settings.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(config.settings.checkpoint_dir + file_name,
                            config.settings.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        assert config.settings.mode in ['train', 'test', 'validate']
        try:
            if config.settings.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, config.settings.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            ssim_score, valid_loss = self.validate()
            self.scheduler.step()
            is_best = ssim_score > self.best_metric
            if is_best:
                self.best_metric = ssim_score
            if epoch % config.settings.save_model_step == 0:
                self.save_checkpoint(file_name=str(epoch) + '_' + config.settings.checkpoint_file,  is_best=is_best)
            self.save_checkpoint(config.settings.checkpoint_file,  is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))
        mean_ssim = 0.0
        # Set the model to be in training mode (for batchnorm)
        self.model.train()
        # Initialize your average meters
        epoch_loss = AverageMeter()

        for images in tqdm_batch:
            if self.cuda:
                images = images.to(self.device, dtype=torch.float32)
            # images = Variable(images)
            self.optimizer.zero_grad()
            # model
            out_images = self.model(images)
            # loss
            cur_loss = self.loss(out_images, images)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            cur_loss.backward()
            self.optimizer.step()

            epoch_loss.update(cur_loss.item())
            # must with no grad or the memory will increase like crazy
            with torch.no_grad():
                mean_ssim += self.metrics(out_images, images, data_range=1.0, size_average=True)

            self.current_iteration += 1
            # exit(0)

        mean_ssim = mean_ssim / len(self.data_loader.train_loader)
        mean_ssim = mean_ssim.detach().cpu().numpy()
        self.summary_writer.add_scalar("epoch-training/loss", epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/mean_ssim", mean_ssim, self.current_iteration)
        tqdm_batch.close()

        logger.info("Training Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + "- mean_ssim: " + str(mean_ssim))

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations, desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in validate mode
        self.model.eval()

        epoch_loss = AverageMeter()
        mean_ssim = 0.0

        with torch.no_grad():
            for images in tqdm_batch:
                if self.cuda:
                    images = images.to(self.device, dtype=torch.float32)
                # model
                out_images = self.model(images)
                # loss
                cur_loss = self.loss(out_images, images)

                if np.isnan(float(cur_loss.item())):
                    raise ValueError('Loss is nan during Validation.')

                mean_ssim += self.metrics(out_images, images, data_range=1.0, size_average=True)
                epoch_loss.update(cur_loss.item())
                # save the reconstructed image
                out_images = out_images.squeeze().detach().cpu().numpy()
                images = images.squeeze().detach().cpu().numpy()
                self.save_output_images(out_images[0:config.settings.number_of_save_images], images[0:config.settings.number_of_save_images])

            mean_ssim = mean_ssim / len(self.data_loader.valid_loader)
            mean_ssim = mean_ssim.detach().cpu().numpy()

            self.summary_writer.add_scalar("epoch_validation/loss", epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_validation/mean_ssim", mean_ssim, self.current_iteration)

            logger.info("Validation Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
                epoch_loss.val) + "- mean_ssim: " + str(mean_ssim))

            tqdm_batch.close()

        return mean_ssim, epoch_loss.val

    def test(self):
        tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations, desc="test at -{}-".format(self.current_epoch))
        self.model.eval()
        mean_ssim = 0.0

        with torch.no_grad():
            for images, masks in tqdm_batch:
                if self.cuda:
                    images = images.to(self.device, dtype=torch.float32)
                # model
                out_images = self.model(images)
                mean_ssim += self.metrics(out_images, images, data_range=1.0, size_average=True)

                # save the reconstructed image
                out_images = out_images.squeeze().detach().cpu().numpy()
                images = images.squeeze().detach().cpu().numpy()
                masks = masks.squeeze().detach().cpu().numpy()
                self.save_output_images(out_images, images, masks=masks)

            mean_ssim = mean_ssim / len(self.data_loader.valid_loader)
            mean_ssim = mean_ssim.detach().cpu().numpy()

            self.summary_writer.add_scalar("epoch_validation/mean_ssim", mean_ssim, self.current_iteration)
            logger.info("Validation Results at epoch-" + str(self.current_epoch) + " | " + ' mean_ssim: ' + str(mean_ssim))
            tqdm_batch.close()

        return mean_ssim

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        logger.info("Please wait while finalizing the operation.. Thank you")
        # self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.settings.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()

    def save_output_images(self, out_images, images, masks=None):
        """helper function to save recunstructed images

        Args:
            out_images (numpy array): a numpy array represents the reconstruction images. shape: N x H x W
            images (numpy array): a numpy array represents the images. shape: N x H x W
            paths (string or Path): image paths
        """
        root = Path(config.settings.out_dir)
        if config.settings.mode == 'train' or config.settings.mode == 'validate':
            root = root / 'validate'
            create_dirs([root])
            out_images = to_uint8(out_images)
            images = to_uint8(images)
            paths = [root / ('epoch' + '_' + str(self.current_epoch) + '_recon_' + str(index) + '.png') for index in range(images.shape[0])]
            save_images(out_images, paths)
            paths = [root / ('epoch' + '_' + str(self.current_epoch) + '_input_' + str(index) + '.png') for index in range(images.shape[0])]
            save_images(images, paths)
        else:
            root = root / 'test'
            create_dirs([root / 'heatmap', root / 'recon', root / 'input', root / 'diff', root / 'mask'])

            # calculate heatmap
            paths = [root / 'heatmap' / (str(index) + '.png') for index in range(images.shape[0])]
            make_heatmaps(output_images=out_images, images=images, paths=paths)

            paths = [root / 'recon' / (str(index) + '.png') for index in range(images.shape[0])]
            out_images = to_uint8(out_images)
            save_images(out_images, paths)
            paths = [root / 'input' / (str(index) + '.png') for index in range(images.shape[0])]
            images = to_uint8(images)
            save_images(images, paths)

            paths = [root / 'diff' / (str(index) + '.png') for index in range(images.shape[0])]
            diff_image = images - out_images
            diff_image = to_uint8(diff_image)
            save_images(diff_image, paths)

            paths = [root / 'mask' / (str(index) + '.png') for index in range(images.shape[0])]
            masks = to_uint8(masks)
            save_images(masks, paths)
