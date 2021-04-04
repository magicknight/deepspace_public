import torch
from torch.optim import lr_scheduler

from deepspace.agents.defect.base import BaseAgent
from deepspace.graphs.models.restoration.dncnn import DnCNN_cheap
from deepspace.datasets.defect.defect import DefectDataLoader
from deepspace.graphs.weights_initializer import xavier_weights
from deepspace.graphs.losses.ssim import SSIM_Loss, ssim, MS_SSIM_Loss, ms_ssim

from commontools.setup import config, logger


class DnCNNAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        # define models
        self.model = DnCNN_cheap(channels=config.deepspace.image_channels, num_of_layers=config.deepspace.number_of_layers)
        self.model = self.model.to(self.device)

        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.deepspace.learning_rate if 'learning_rate' in config.deepspace else 1e-3,
            betas=config.deepspace.betas if 'betas' in config.deepspace else [0.9, 0.999],
            eps=config.deepspace.eps if 'eps' in config.deepspace else 1e-8,
            weight_decay=config.deepspace.weight_decay if 'weight_decay' in config.deepspace else 1e-5)

        # Define Scheduler
        def lambda1(epoch): return pow(1 - epoch / config.deepspace.max_epoch, 0.9)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        self.model.apply(xavier_weights)

        # define data_loader
        self.data_loader = DefectDataLoader()

        # define loss
        self.loss = MS_SSIM_Loss(channel=config.deepspace.image_channels, data_range=1.0, size_average=True)
        self.loss = self.loss.to(self.device)

        # define metrics
        self.metrics = ms_ssim

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint()
