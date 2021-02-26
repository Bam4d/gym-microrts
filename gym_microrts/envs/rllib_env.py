from typing import List, Tuple

import gym
import torch
from gym.spaces import Dict, MultiDiscrete
from ray.rllib import MultiAgentEnv, VectorEnv
import numpy as np
from ray.rllib.utils.typing import EnvActionType, EnvObsType, EnvInfoDict

from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


class RllibMicroRTSWrapper(MicroRTSGridModeVecEnv, VectorEnv):

    def __init__(self, env_config):
        super().__init__(**env_config)

        self._initialize_spaces()

        VectorEnv.__init__(self, self.observation_space, self.action_space, env_config['num_bot_envs'])

    def _initialize_spaces(self):
        raise NotImplementedError()

    def _transform(self, observation):
        return observation.astype(np.float)

    def get_masks(self):
        invalid_action_masks = np.array(self.vec_client.getMasks(0))

        self._valid_locations = invalid_action_masks[:, :, :, 0].reshape(self.num_envs, -1)

        # Cannot have masks with all zeros, must create nop masks that force the distribution to choose a nop value
        invalid_action_masks = invalid_action_masks[:, :, :, 1:]
        flattened_masks = invalid_action_masks.reshape(self.num_envs, -1)

        for split in np.split(flattened_masks, np.cumsum(self.action_space.nvec), axis=1):
            split[np.where(np.sum(split, axis=1) == 0)[0], :] = 1

        return flattened_masks

    def vector_reset(self, **kwargs):
        obs_batch = super().reset(**kwargs)
        masks_batch = self.get_masks()
        return [{
            'invalid_action_masks': masks,
            'obs': obs
        } for masks, obs in zip(masks_batch, obs_batch)]

    def reset_at(self, index: int) -> EnvObsType:
        pass

    def vector_step(self, actions_batch):
        filtered_actions = []
        for actions, location_mask in zip(actions_batch, self._valid_locations.tolist()):
            locations = np.where(location_mask)[0]
            filtered_actions.append(np.hstack([np.expand_dims(locations.T, 1), actions.reshape(-1, 7)[locations]]))

        obs_batch, reward_batch, done_batch, infos_batch = super().step(filtered_actions)

        masks_batch = self.get_masks()

        obs = [{
            'invalid_action_masks': masks,
            'obs': obs
        } for masks, obs in zip(masks_batch, obs_batch)]

        return obs, reward_batch.tolist(), done_batch.tolist(), infos_batch


class GridnetWrapper(RllibMicroRTSWrapper):

    def __init__(self, env_config):
        super().__init__(env_config)

    def _initialize_spaces(self):
        num_grid_locations = self.action_space.nvec[0]
        cell_discrete_action_shape = self.action_space.nvec[1:]

        self.num_action_logits = np.sum(cell_discrete_action_shape)

        cell_multi_discretes = []
        for g in range(num_grid_locations):
            cell_multi_discretes.extend(cell_discrete_action_shape)

        self.height = self.observation_space.shape[0]
        self.width = self.observation_space.shape[1]

        self.action_space = MultiDiscrete(cell_multi_discretes)

        self.observation_space = Dict({
            'obs': gym.spaces.Box(
                self.observation_space.low,
                self.observation_space.high,
                dtype=np.float
            ),
            'invalid_action_masks': gym.spaces.Box(
                0.0,
                1.0,
                shape=[self.height * self.width * self.num_action_logits],
                dtype=np.float
            )
        })
