"""
Main agent for DQN
"""
import math
import random

import gym
import torch
from tensorboardX import SummaryWriter
# from torch.backends import cudnn
from tqdm import tqdm

from deepspace.agents.base import BasicAgent
from deepspace.graphs.losses.huber_loss import HuberLoss
from deepspace.graphs.models.reinforcement.dqn import DQN
from deepspace.utils.replay_memory import ReplayMemory, Transition
from commontools.setup import config, logger

# cudnn.benchmark = True


class DQNAgent(BasicAgent):

    def __init__(self):
        super().__init__()
        config.swap.device = self.device
        # define models (policy and target)
        self.policy_model = DQN()
        self.target_model = DQN()
        # define memory
        self.memory = ReplayMemory(config.deepspace)
        # define loss
        self.loss = HuberLoss()
        # define optimizer
        self.optimizer = torch.optim.RMSprop(self.policy_model.parameters())

        # define environment
        self.env = gym.make('deepspace.environments:detection-v0', max_projections=config.dxray.max_step, resolution=tuple(config.dxray.resolution)).unwrapped

        # initialize counter
        self.episode_durations = []

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint()

        self.policy_model = self.policy_model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.loss = self.loss.to(self.device)

        # Initialize Target model with policy model state dict
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=config.swap.summary_dir, comment='DQN')

        # save checkpoint
        self.checkpoint = {
            'current_episode': self.current_episode,
            'current_iteration': self.current_iteration,
            'policy_model.state_dict': self.policy_model.state_dict(),
            'optimizer.state_dict': self.optimizer.state_dict(),
            # 'number': {
            #     'current_episode': self.current_episode,
            #     'current_iteration': self.current_iteration,
            # },
            # 'state_dict': {
            #     'policy_model': self.policy_model,
            #     'optimizer': self.optimizer,
            # },
        }

    def select_action(self, state):
        """
        The action selection function, it either uses the model to choose an action or samples one uniformly.
        :param state: current state of the model
        :return:
        """
        state = state.to(self.device)
        sample = random.random()
        eps_threshold = config.deepspace.eps_start + (config.deepspace.eps_start - config.deepspace.eps_end) * math.exp(
            -1. * self.current_iteration / config.deepspace.eps_decay)
        self.current_iteration += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimize_policy_model(self):
        """
        performs a single step of optimization for the policy model
        :return:
        """
        if self.memory.length() < config.deepspace.batch_size:
            return
        # sample a batch
        transitions = self.memory.sample_batch(config.deepspace.batch_size)

        one_batch = Transition(*zip(*transitions))

        # create a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, one_batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in one_batch.next_state if s is not None])

        # concatenate all batch elements into one
        state_batch = torch.cat(one_batch.state)
        action_batch = torch.cat(one_batch.action)
        reward_batch = torch.cat(one_batch.reward)

        state_batch = state_batch.to(self.device)
        non_final_next_states = non_final_next_states.to(self.device)

        curr_state_values = self.policy_model(state_batch)
        curr_state_action_values = curr_state_values.gather(1, action_batch)

        next_state_values = torch.zeros(config.deepspace.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        # Get the expected Q values
        expected_state_action_values = (next_state_values * config.deepspace.gamma) + reward_batch
        # compute loss: temporal difference error
        loss = self.loss(curr_state_action_values, expected_state_action_values.unsqueeze(1))

        # optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def train(self):
        """
        Training loop based on the number of episodes
        :return:
        """
        for episode in tqdm(range(self.current_episode, config.deepspace.num_episodes)):
            self.current_episode = episode
            # reset environment
            self.env.reset()
            self.train_one_epoch()
            # The target network has its weights kept frozen most of the time
            if self.current_episode % config.deepspace.target_update == 0:
                self.target_model.load_state_dict(self.policy_model.state_dict())

        self.env.render()
        self.env.close()
        self.save_checkpoint(config.deepspace.checkpoint_file)

    def train_one_epoch(self):
        """
        One episode of training; it samples an action, observe next screen and optimize the model once
        :return:
        """
        episode_duration = 0
        prev_frame = self.cartpole.get_screen(self.env)
        curr_frame = self.cartpole.get_screen(self.env)
        # get state
        curr_state = curr_frame - prev_frame

        while(1):
            episode_duration += 1
            # select action
            action = self.select_action(curr_state)
            # perform action and get reward
            _, reward, done, _ = self.env.step(action.item())

            reward = torch.Tensor([reward]).to(self.device)

            prev_frame = curr_frame
            curr_frame = self.cartpole.get_screen(self.env)
            # assign next state
            if done:
                next_state = None
            else:
                next_state = curr_frame - prev_frame

            # add this transition into memory
            self.memory.push_transition(curr_state, action, next_state, reward)

            curr_state = next_state

            # Policy model optimization step
            curr_loss = self.optimize_policy_model()
            if curr_loss is not None:
                curr_loss = curr_loss.cpu()
                self.summary_writer.add_scalar("Temporal_Difference_Loss", curr_loss.detach().numpy(), self.current_iteration)
            # check if done
            if done:
                break

        self.summary_writer.add_scalar("Training_Episode_Duration", episode_duration, self.current_episode)

    def validate(self):
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        logger.info("Please wait while finalizing the operation.. Thank you")
        # self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(config.swap.summary_dir))
        self.summary_writer.close()
