"""
Defect detection agent
"""
import numpy as np

from tqdm import tqdm
import shutil
from pathlib import Path

import torch
from torch.backends import cudnn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from deepspace.agents.base import BaseAgent

from deepspace.graphs.models.autoencoder import AutoEncoder, AnomalyAE
from deepspace.datasets.defect.defect import DefectDataLoader
from deepspace.graphs.losses.ssim import SSIM_Loss, ssim, MS_SSIM_Loss, ms_ssim
from deepspace.graphs.losses.mse import MaskLoss

from deepspace.utils.metrics import AverageMeter, AverageMeterList
from deepspace.utils.data import to_uint8, save_images, make_heatmaps
from deepspace.utils.dirs import create_dirs
from deepspace.utils.train_utils import get_device
from deepspace.config.config import config, logger

cudnn.benchmark = True


class BaseAgent(BaseAgent):

    def __init__(self):
        super().__init__()

        # set cuda flag
        self.device = get_device()
        config.swap.device = self.device

        # define models
        # self.model = AutoEncoder(
        #     C=config.settings.model_c if 'model_c' in config.settings else 128,
        #     M=config.settings.model_m if 'model_m' in config.settings else 64,
        #     in_chan=config.settings.image_channels,
        #     out_chan=config.settings.image_channels)
        # self.model = self.model.to(self.device)

        # define data_loader
        self.data_loader = DefectDataLoader()

        # define loss
        self.loss = SSIM_Loss(channel=config.settings.image_channels, data_range=1.0, size_average=True)
        self.loss = self.loss.to(self.device)

        # define metrics
        self.metrics = ssim

        # Create instance from the optimizer
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=config.settings.learning_rate if 'learning_rate' in config.settings else 1e-3,
        #     betas=config.settings.betas if 'betas' in config.settings else [0.9, 0.999],
        #     eps=config.settings.eps if 'eps' in config.settings else 1e-8,
        #     weight_decay=config.settings.weight_decay if 'weight_decay' in config.settings else 1e-5)

        # Define Scheduler
        # def lambda1(epoch): return pow(1 - epoch / config.settings.max_epoch, 0.9)
        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=config.swap.summary_dir, comment='AutoEncoder')

    def load_checkpoint(self):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        file_name = config.swap.checkpoint_dir / config.settings.checkpoint_file
        try:
            logger.info("Loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                        .format(config.swap.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            logger.info("No checkpoint exists from '{}'. Skipping...".format(config.swap.checkpoint_dir))
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
        torch.save(state, config.swap.checkpoint_dir / file_name)
        # backup model on a certain steps
        if self.current_epoch % config.settings.save_model_step == 0:
            torch.save(state, config.swap.checkpoint_dir / (str(self.current_epoch) + '_' + config.settings.checkpoint_file))
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(
                config.swap.checkpoint_dir / file_name,
                config.swap.checkpoint_dir / 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        assert config.settings.mode in ['train', 'test']
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
        for epoch in tqdm(range(self.current_epoch, config.settings.max_epoch), desc="traing at -{}-, total epoches -{}-".format(self.current_epoch, config.settings.max_epoch)):
            self.current_epoch = epoch
            self.train_one_epoch()
            ssim_score = self.validate()
            self.scheduler.step()
            # self.scheduler.step(np.round_(ssim_score, 4))
            is_best = ssim_score > self.best_metric
            if is_best:
                self.best_metric = ssim_score
            self.save_checkpoint(config.settings.checkpoint_file,  is_best=is_best)

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
        mean_ssim = 0.0
        # Set the model to be in training mode (for batchnorm)
        self.model.train()
        # Initialize average meters
        epoch_loss = AverageMeter()
        # loop images
        for defect_images, normal_images in tqdm_batch:
            defect_images = defect_images.to(self.device, dtype=torch.float32)
            normal_images = normal_images.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            # model
            out_images = self.model(defect_images)
            # loss
            cur_loss = self.loss(out_images, normal_images)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            # optimizer
            cur_loss.backward()
            self.optimizer.step()
            epoch_loss.update(cur_loss.item())
            # must with no grad here, or the mean_ssim and images will be keeped in the memory and the memory will increase like crazy
            with torch.no_grad():
                mean_ssim += self.metrics(out_images, normal_images, data_range=1.0, size_average=True)
            self.current_iteration += 1

        # logging
        mean_ssim = mean_ssim / len(self.data_loader.train_loader)
        mean_ssim = mean_ssim.detach().cpu().numpy()
        self.summary_writer.add_scalar("epoch-training/loss", epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/mean_ssim", mean_ssim, self.current_iteration)
        tqdm_batch.close()
        logger.info(
            "Training Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
                epoch_loss.val) + "- mean_ssim: " + str(mean_ssim))

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations, desc="Valiation at -{}-".format(self.current_epoch))
        # set the model in validate mode
        self.model.eval()
        # initialize
        mean_ssim = 0.0
        # loop validation images
        with torch.no_grad():
            for defect_images, normal_images in tqdm_batch:
                defect_images = defect_images.to(self.device, dtype=torch.float32)
                normal_images = normal_images.to(self.device, dtype=torch.float32)
                # model
                out_images = self.model(defect_images)
                mean_ssim += self.metrics(out_images, normal_images, data_range=1.0, size_average=True)
                # save the reconstructed image
                out_images = out_images.squeeze().detach().cpu().numpy()
                defect_images = defect_images.squeeze().detach().cpu().numpy()
                normal_images = normal_images.squeeze().detach().cpu().numpy()

                self.save_output_images(out_images[config.settings.save_images[0]: config.settings.save_images[1]],
                                        defect_images[config.settings.save_images[0]: config.settings.save_images[1]],
                                        normal_images[config.settings.save_images[0]: config.settings.save_images[1]])
            # logging
            mean_ssim = mean_ssim / len(self.data_loader.valid_loader)
            mean_ssim = mean_ssim.detach().cpu().numpy()
            self.summary_writer.add_scalar("epoch_validation/mean_ssim", mean_ssim, self.current_iteration)
            logger.info("Validation Results at epoch-" + str(self.current_epoch) + " | " + "- mean_ssim: " + str(mean_ssim))
            tqdm_batch.close()
        return mean_ssim

    def test(self):
        tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations, desc="test at -{}-".format(self.current_epoch))
        self.model.eval()
        mean_ssim = 0.0
        with torch.no_grad():
            for defect_images, normal_images, ground_truth_images in tqdm_batch:
                defect_images = defect_images.to(self.device, dtype=torch.float32)
                normal_images = normal_images.to(self.device, dtype=torch.float32)
                ground_truth_images = ground_truth_images.to(self.device, dtype=torch.float32)
                # model
                out_images = self.model(defect_images)
                mean_ssim += self.metrics(out_images, normal_images, data_range=1.0, size_average=True)
                # save the reconstructed image
                out_images = out_images.squeeze().detach().cpu().numpy()
                defect_images = defect_images.squeeze().detach().cpu().numpy()
                normal_images = normal_images.squeeze().detach().cpu().numpy()
                ground_truth_images = ground_truth_images.squeeze().detach().cpu().numpy()
                self.save_output_images(out_images, defect_images, normal_images, ground_truth_images)
            # logging
            mean_ssim = mean_ssim / len(self.data_loader.test_loader)
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
        self.summary_writer.export_scalars_to_json(config.swap.summary_dir / "all_scalars.json")
        self.summary_writer.close()
        self.data_loader.finalize()

    def save_output_images(self, out_images, defect_images=None, normal_images=None, ground_truth_images=None):
        """helper function to save recunstructed images

        Args:
            out_images (numpy array): a numpy array represents the reconstruction images. shape: N x H x W
            defect_images (numpy array): a numpy array represents the defect_images. shape: N x H x W
            normal_images (numpy array): a numpy array represents the normal_images. shape: N x H x W
            ground_truth_images (numpy array): a numpy array represents the ground_truth_images. shape: N x H x W
        """
        try:
            root = Path(config.swap.out_dir)
            if config.settings.mode == 'train':
                root = root / 'validate'
                create_dirs([root])
                # only save noraml and defect images on epoch 0, since they dont change each epoch
                if self.current_epoch == 0:
                    defect_images = to_uint8(defect_images)
                    normal_images = to_uint8(normal_images)
                    paths = [root / ('epoch' + '_' + str(self.current_epoch) + '_input_' + str(index) + '.png') for index in range(defect_images.shape[0])]
                    save_images(defect_images, paths)
                    paths = [root / ('epoch' + '_' + str(self.current_epoch) + '_normal_' + str(index) + '.png') for index in range(normal_images.shape[0])]
                    save_images(normal_images, paths)
                out_images = to_uint8(out_images)
                paths = [root / ('epoch' + '_' + str(self.current_epoch) + '_recon_' + str(index) + '.png') for index in range(out_images.shape[0])]
                save_images(out_images, paths)

            else:
                root = root / 'test'
                create_dirs([root / 'heatmap', root / 'recon', root / 'input', root / 'normal', root / 'diff', root / 'ground_truth'])
                # calculate heatmap
                paths = [root / 'heatmap' / (str(index) + '.png') for index in range(defect_images.shape[0])]
                make_heatmaps(output_images=out_images, images=defect_images, paths=paths)

                paths = [root / 'recon' / (str(index) + '.png') for index in range(out_images.shape[0])]
                out_images = to_uint8(out_images)
                save_images(out_images, paths)
                paths = [root / 'input' / (str(index) + '.png') for index in range(defect_images.shape[0])]
                defect_images = to_uint8(defect_images)
                save_images(defect_images, paths)
                paths = [root / 'normal' / (str(index) + '.png') for index in range(normal_images.shape[0])]
                normal_images = to_uint8(normal_images)
                save_images(normal_images, paths)

                paths = [root / 'diff' / (str(index) + '.png') for index in range(defect_images.shape[0])]
                diff_image = defect_images - out_images
                diff_image = to_uint8(diff_image)
                save_images(diff_image, paths)

                paths = [root / 'ground_truth' / (str(index) + '.png') for index in range(ground_truth_images.shape[0])]
                ground_truth_images = to_uint8(ground_truth_images)
                save_images(ground_truth_images, paths)
        except Exception as e:
            logger.error(e)
