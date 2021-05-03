"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import torch
from pathlib import Path

from commontools.setup import config, logger
from deepspace.utils.io.io import save_checkpoint, load_checkpoint, save_settings
from deepspace.utils.train_utils import get_device


"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self):
        pass

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError


class BasicAgent(BaseAgent):
    """
    This basic class will contain the basic functions to be overloaded by any agent you will implement.
    """

    def __init__(self):
        super().__init__()
        # set cuda flag
        self.device = get_device()
        self.summary_writer = None
        self.data_loader = None
        self.current_episode = 0
        self.current_iteration = 0
        self.current_epoch = 0
        self.best_metric = 0

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        filename = Path(config.swap.checkpoint_dir) / file_name
        # Save the state
        save_checkpoint(self, filename, is_best)

    def load_checkpoint(self, file_name="checkpoint.pth.tar"):
        filename = Path(config.swap.checkpoint_dir) / file_name
        try:
            logger.info("Loading checkpoint '{}'".format(filename))
            load_checkpoint(self, file_name)
            logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                        .format(config.swap.checkpoint_dir, self.current_episode, self.current_iteration))
        except OSError as e:
            logger.info("No checkpoint exists from '{}'. Skipping...".format(config.swap.checkpoint_dir))
            logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if config.deepspace.mode == 'test':
                self.test()
            elif config.deepspace.mode == 'train':
                self.train()

        except KeyboardInterrupt:
            logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        logger.info("Please wait while finalizing the operation.. Thank you")
        self.summary_writer.export_scalars_to_json(config.swap.summary_dir / "all_scalars.json")
        self.summary_writer.close()
        save_settings(config, config.swap.work_space / 'config.toml')
        self.data_loader.finalize()
