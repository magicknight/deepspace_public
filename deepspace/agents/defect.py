"""
Defect detection agent
"""
import numpy as np

from tqdm import tqdm
import shutil
import random

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
from deepspace.graphs.losses.ssim import SSIM, ssim, MS_SSIM, ms_ssim

from deepspace.utils.metrics import AverageMeter, AverageMeterList
from deepspace.utils.misc import print_cuda_statistics
from deepspace.config.config import config, logger

cudnn.benchmark = True


class DefectAgent(BaseAgent):

    def __init__(self):
        super().__init__(config)

        # define models
        self.model = AutoEncoder(in_chan=config.settings.image_channels, out_chan=config.settings.image_channels)

        # define data_loader
        self.data_loader = DefectDataLoader()

        # define loss
        self.loss = SSIM(channel=config.settings.image_channels)

        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config.settings.learning_rate,
                                          betas=(config.settings.betas[0], config.settings.betas[1]),
                                          eps=config.settings.eps,
                                          weight_decay=config.settings.weight_decay)

        # Define Scheduler
        def lambda1(epoch): return pow((1 - ((epoch - 1) / config.settings.max_epoch)), 0.9)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

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

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

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
            self.scheduler.step(epoch)
            self.train_one_epoch()

            ssim_score, valid_loss = self.validate()
            self.scheduler.step(valid_loss)

            is_best = ssim_score > self.best_metric
            if is_best:
                self.best_metric = ssim_score

            self.save_checkpoint(is_best=is_best)

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
        metrics = ssim if config.settings.loss == 'ssim' else ms_ssim

        for images in tqdm_batch:
            if self.cuda:
                images = images.pin_memory().cuda()
            images = Variable(images)
            # model
            pred = self.model(images)
            # loss
            cur_loss = self.loss(pred, images)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            epoch_loss.update(cur_loss.item())
            mean_ssim += metrics(pred, images)

            self.current_iteration += 1
            # exit(0)

        mean_ssim /= len(self.data_loader.train_loader.dataset)
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
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        metrics = ssim if config.loss == 'ssim' else ms_ssim
        mean_ssim = 0.0

        with torch.no_grad():
            for index, images in enumerate(tqdm_batch):
                if self.cuda:
                    images = images.pin_memory().cuda()
                images = Variable(images)
                # model
                pred = self.model(images)
                # if index == 20:
                #     imageio((outputs*255).squeeze(0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)).save('recons_%s.png' % (opts.loss_type))
                # loss
                cur_loss = self.loss(pred, images)

                if np.isnan(float(cur_loss.item())):
                    raise ValueError('Loss is nan during Validation.')

                mean_ssim += metrics(pred, images)
                epoch_loss.update(cur_loss.item())

            mean_ssim /= len(self.data_loader.valid_loader.dataset)

            self.summary_writer.add_scalar("epoch_validation/loss", epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch_validation/mean_ssim", mean_ssim, self.current_iteration)

            logger.info("Validation Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
                epoch_loss.val) + "- mean_ssim: " + str(mean_ssim))

            tqdm_batch.close()

        return mean_ssim, epoch_loss.val

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
