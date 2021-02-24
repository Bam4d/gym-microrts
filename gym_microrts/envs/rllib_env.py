import gym
import torch
from gym.spaces import Tuple, Dict, MultiDiscrete
from ray.rllib import MultiAgentEnv
import numpy as np

from gym_microrts.envs.vec_env import MicroRTSVecEnv


class RllibMicroRTSWrapper(MicroRTSVecEnv):

    def __init__(self, env_config):
        super().__init__(**env_config)

        self._initialize_spaces()

    def _initialize_spaces(self):
        raise NotImplementedError()

    def _transform(self, observation):
        return observation.astype(np.float)

    def get_masks(self):
        invalid_action_masks = np.array(self.vec_client.getMasks(0))
        return invalid_action_masks.reshape(-1, invalid_action_masks.shape[-1])

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs = {
            'invalid_action_masks': self.get_masks(),
            'obs': obs
        }
        return obs

    def step(self, action_dict):
        actions_array = np.zeros((self.player_count, *self.action_space.shape))
        for agent_id, action in action_dict.items():
            actions_array[agent_id - 1] = action

        obs, reward, done, infos = super().step(actions_array)

        obs = {
            'invalid_action_masks': self.get_masks(),
            'obs': obs
        }

        return obs, reward, done, infos


class GridnetWrapper(RllibMicroRTSWrapper):

    def __init__(self, env_config):
        super().__init__(env_config)


    def _initialize_spaces(self):
        num_grid_locations = self.action_space.nvec[0]
        num_cell_outputs = self.action_space.nvec[1:]

        cell_multi_discretes = []
        for g in range(num_grid_locations):
            cell_multi_discretes.extend(num_cell_outputs)

        self.action_space = MultiDiscrete(cell_multi_discretes)

        self.height = self.observation_space.shape[0]
        self.width = self.observation_space.shape[1]

        self.observation_space = Dict({
            'obs': gym.spaces.Box(
                self.observation_space.low,
                self.observation_space.high,
                dtype=np.float
            ),
            'invalid_action_masks': self.action_space
        })


class GridnetSelfPlayWrapper(GridnetWrapper, MultiAgentEnv):

    def _to_multi_agent_map(self, data):
        return {i: self._transform(data) for i, data in enumerate(data)}

    def reset(self):
        obs = super().reset()
        return self._to_multi_agent_map(obs)

    def step(self, actions):
        obs, reward, done, infos = super().step(actions)
        return self._to_multi_agent_map(obs), self._to_multi_agent_map(reward), self._to_multi_agent_map(
            done), self._to_multi_agent_map(infos)
