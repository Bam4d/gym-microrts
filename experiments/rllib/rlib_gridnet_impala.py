import numpy as np
import ray
from ray import tune
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.impala.vtrace_torch_policy import VTraceTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env

import torch
from torch import nn

from gym_microrts import microrts_ai
from gym_microrts.envs.rllib_env import GridnetWrapper
from gym_microrts.rllib.models.common import Transpose, layer_init, Reshape


# def gridnet_action_sampler(policy, model, input_dict, state_batches, explore, timestep):
#     """
#     Given our action predictions
#     """
#     actions, logp, state_out
#
# GridnetVTraceTorchPolicy = VTraceTorchPolicy.with_updates(
#     name="GridnetVTraceTorchPolicy",
#     action_sampler_fn=gridnet_action_sampler,
# )

# def get_policy_class(config):
#     if config['framework'] == 'torch':
#         return GridnetVTraceTorchPolicy
#     else:
#         raise NotImplementedError('Tensorflow not supported')
#
#
# GridnetImpalaTrainer = ImpalaTrainer.with_updates(default_policy=GridnetVTraceTorchPolicy,
#                                                   get_policy_class=get_policy_class)

class GridnetAgent(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        height = obs_space.original_space['obs'].shape[0]
        width = obs_space.original_space['obs'].shape[1]

        grid_channels = 78

        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),  # "bhwc" -> "bchw"
            layer_init(nn.Conv2d(27, 32, kernel_size=5, padding=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=5, padding=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(64, grid_channels, kernel_size=1)),
            Transpose((0, 3, 1, 2)),
            Reshape((-1, height * width * grid_channels))
        )

        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * height * width, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1))

    def forward(self, input_dict, state, seq_lens):
        input_obs = input_dict['obs']['obs']
        input_mask = input_dict['obs']['invalid_action_masks']
        self._encoded = self.encoder(input_obs)

        # Value function
        value = self.critic(self._encoded)
        self._value = value.reshape(-1)

        # Logits for actions
        logits = self.actor(self._encoded)

        masked_logits = logits + torch.log(input_mask)

        return masked_logits, state

    def value_function(self):
        return self._value


if __name__ == '__main__':
    # sep = os.pathsep
    # os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(local_mode=True)

    env_name = 'ray-microrts-env'

    register_env(env_name, GridnetWrapper)
    ModelCatalog.register_custom_model('GridnetAgent', GridnetAgent)

    config = {
        'framework': 'torch',
        'num_workers': 1,
        'num_envs_per_worker': 1,

        'model': {
            'custom_model': 'GridnetAgent',
            'custom_model_config': {},
        },
        'env': env_name,
        'env_config': {
            'num_envs': 1,
            'max_steps': 2000,
            'render_theme': 1,
            'frame_skip': 0,
            'ai2s': [microrts_ai.passiveAI],
            'map_path': 'maps/16x16/basesWorkers16x16A.xml',
            'reward_weight': np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        },

        # 'lr': tune.grid_search([0.0001, 0.0005, 0.001, 0.005])
    }

    stop = {
        'timesteps_total': 10000,
    }

    result = tune.run(ImpalaTrainer, config=config, stop=stop)
