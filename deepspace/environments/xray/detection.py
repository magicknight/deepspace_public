"""
X-Ray evironment for deep reinforcemnet
"""
import torch

import gym
from gym import logger, spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from deepspace.graphs.models.gan.dcgan import Generator
from deepspace.utils.train_utils import get_device
from deepspace.environments.xray.xray import Astra
from deepspace.config.config import config, logger

matplotlib.rcParams['toolbar'] = 'None'


class Detection(gym.Env):
    metadata = {
        'render.modes': ['human', 'array']
    }

    def __init__(self, max_projections=3, resolution=(512, 512)):
        self.current_step = 0
        self.projector = Astra(max_projections, resolution)
        # do the projection
        self.projector.project(self.angle)
        # Initial state (can be reset later)
        self.detected = False
        self.state = (self.projector.projections, self.detected)

        self.reward_range = (0, 10)
        angle_range = np.array(config.environment.angle_range)
        # self.action_space = spaces.Box(low=angle_range[:, 0], high=angle_range[:, 1], )  # in real life the parameter selection model may not be able to select exactly angles.
        self.action_space = spaces.Box(low=angle_range[2, 0], high=angle_range[2, 1], shape=(1, ))  # select rotate x degree from current position
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.projector.projections.shape, dtype=np.float)  # observations
        self.viewer = rendering.SimpleImageViewer(display=None, maxwidth=config.environment.resolution[0])

        self.device = get_device()
        self.model = Generator()
        self.model = self.model.to(self.device)
        self.load_model()
        self.model.eval()

    def seed(self, seed=None):  # pragma: no cover
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # update detect status

        reward = self.draw_click(action)
        if reward == 1:
            clicks += 1
            ads[action].clicks += 1

        # Update impressions
        ads[action].impressions += 1
        impressions += 1

        # Update the ctr time series (for rendering)
        if impressions % self.time_series_frequency == 0:
            ctr = 0.0 if impressions == 0 else float(clicks / impressions)
            self.ctr_time_series.append(ctr)

        self.state = (ads, impressions, clicks)

        return self.state, reward, False, {}

    def reset(self):
        # reset image arrays to 0
        self.projector.reset()
        self.detected = False
        self.state = (self.projector.projections, self.detected)
        self.angle = self.init_angles()
        self.current_step = 0
        return self.state

    def render(self, mode='human', angle=None):
        if mode == 'human':
            self.viewer.imshow(self.projector.projections[self.current_step])

    def draw_click(self, action):
        if self.reward_policy is not None:
            return self.reward_policy(action)

        if self.click_probabilities is None:
            self.click_probabilities = [self.np_random.uniform() * 0.5 for i in range(self.num_ads)]

        return 1 if self.np_random.uniform() <= self.click_probabilities[action] else 0

    def load_model(self) -> None:
        """
        load detection model
        :return:
        """
        try:
            logger.info("Loading checkpoint '{}'".format(config.environment.model_path))
            checkpoint = torch.load(config.environment.model_path)
            self.model.load_state_dict(checkpoint['gen_state_dict'])
            logger.info("Checkpoint loaded successfully at (epoch {}) at (iteration {})\n"
                        .format(checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            logger.info("No checkpoint exists from '{}'. Skipping...".format(config.environment.model_path))

    def detect(self) -> None:
        test_image = self.projector.projections[self.current_step]
        # defect detection
        fake_image = self.generator(test_image)
        zero_array = torch.zeros_like(test_image)
        one_array = torch.ones_like(test_image)
        diff_image = test_image - fake_image
        diff_image = torch.where(diff_image > config.environment.image_threshold[1], zero_array, diff_image)
        diff_image = torch.where(diff_image > config.environment.image_threshold[0], one_array, zero_array)
        diff_image = diff_image.squeeze()
        detect_area = torch.sum(diff_image, (0, 1))
        self.detected = detect_area > config.environment.area_threshold

    def close(self):
        plt.close()
