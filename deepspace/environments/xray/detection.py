"""
detection evironment for deep reinforcemnet
"""
import torch
from torchvision import transforms

import gym
from gym import logger, spaces
from gym.utils import seeding
# from gym.envs.classic_control import rendering

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from deepspace.graphs.models.gan.dcgan import Generator
from deepspace.utils.train_utils import get_device
from deepspace.environments.xray.xray import Astra
from commontools.setup import config, logger

matplotlib.rcParams['toolbar'] = 'None'


class Detection(gym.Env):
    metadata = {
        'render.modes': ['human', 'array']
    }

    def __init__(self, max_projections=3, resolution=(512, 512)):
        self.max_step = max_projections
        self.resolution = resolution
        self.current_step = 0
        self.projector = Astra(max_projections, resolution)
        # Initial state (can be reset later)
        self.detected = False
        self.state = (self.projector.projections, self.detected)

        self.reward_range = (0, 1)
        self.reward = 0
        angle_range = np.array(config.dxray.angle_range)
        # self.action_space = spaces.Box(low=angle_range[:, 0], high=angle_range[:, 1], )  # in real life the parameter selection model may not be able to select exactly angles.
        self.action_space = spaces.Box(low=angle_range[2, 0], high=angle_range[2, 1], shape=(1, ))  # select rotate x degree from current position
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.projector.projections.shape, dtype=np.float)  # observations
        # self.viewer = rendering.SimpleImageViewer(display=None, maxwidth=config.dxray.resolution[0])

        self.device = get_device()
        self.model = Generator()
        self.model = self.model.to(self.device)
        self.load_model()
        self.model.eval()

        # transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def seed(self, seed=None):  # pragma: no cover
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # change projector angle
        self.projector.rotate(action)
        # projection
        self.projector.project()
        # update detect status
        self.detect()
        # update reward
        self.reward_policy()
        # update step count
        self.step += 1
        # update state
        self.state = (self.projector.projections, self.detected)
        # done or not
        is_done = self.step >= self.max_step
        return self.state, self.reward, is_done, {}

    def reset(self):
        while True:
            # reset image arrays to 0
            self.current_step = 0
            self.projector.reset()
            self.detect()
            if self.detected:
                continue
            else:
                break
        self.detected = False
        self.reward_policy()
        self.state = (self.projector.projections, self.detected)
        return self.state

    def render(self, mode='human', angle=None):
        pass
        # if mode == 'human':
        #     self.viewer.imshow(self.projector.projections[self.current_step])

    def load_model(self) -> None:
        """
        load detection model
        :return:
        """
        try:
            logger.info("Loading checkpoint '{}'".format(config.dxray.model_path))
            checkpoint = torch.load(config.dxray.model_path)
            self.model.load_state_dict(checkpoint['gen_state_dict'])
            logger.info("Checkpoint loaded successfully at (epoch {}) at (iteration {})\n"
                        .format(checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            logger.info("No checkpoint exists from '{}'. Skipping...".format(config.dxray.model_path))

    def detect(self) -> None:
        test_image = self.projector.projections[self.current_step]
        test_image = self.transform(test_image)
        test_image = test_image.unsqueeze(0).float()
        test_image = test_image.to(self.device)
        # defect detection
        fake_image = self.model(test_image)
        zero_array = torch.zeros_like(test_image)
        one_array = torch.ones_like(test_image)
        diff_image = test_image - fake_image
        diff_image = torch.where(diff_image > config.dxray.image_threshold[1], zero_array, diff_image)
        diff_image = torch.where(diff_image > config.dxray.image_threshold[0], one_array, zero_array)
        diff_image = diff_image.squeeze()
        detect_area = torch.sum(diff_image, (0, 1))
        self.detected = detect_area > config.dxray.area_threshold

    def reward_policy(self):
        self.reward = 1 if self.detected else 0

    def close(self):
        plt.close()
