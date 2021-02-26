import numpy as np
import ray
from ray import tune
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.impala.vtrace_torch_policy import VTraceTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical, TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.tune.registry import register_env

import torch
from torch import nn

from gym_microrts import microrts_ai
from gym_microrts.envs.rllib_env import GridnetWrapper
from gym_microrts.rllib.models.common import Transpose, layer_init, Reshape


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
            Transpose((0, 3, 1, 2)),  # "bchw" -> "bhwc"
            nn.Flatten()
        )

        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * height * width, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1))

    def forward(self, input_dict, state, seq_lens):
        input_obs = input_dict['obs']['obs']
        invalid_action_mask = input_dict['obs']['invalid_action_masks']
        self._encoded = self.encoder(input_obs)

        # Value function
        value = self.critic(self._encoded)
        self._value = value.reshape(-1)

        # Logits for actions
        logits = self.actor(self._encoded)

        masked_logits = logits + torch.log(invalid_action_mask)

        return masked_logits, state

    def value_function(self):
        return self._value

class GridnetMultiCategorical(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        input_lens = model.action_space.nvec
        self._flattened_sample_size = len(input_lens)
        self._flattened_logits_size = inputs.shape[1]
        super().__init__(inputs, model)
        # If input_lens is np.ndarray or list, force-make it a tuple.
        inputs_split = self.inputs.reshape(-1, 78).split(tuple(input_lens[:7]),dim=1)
        self.cats = [
            torch.distributions.categorical.Categorical(logits=input_)
            for input_ in inputs_split
        ]

    def sample(self):
        arr = []
        for i, cat in enumerate(self.cats):
            if cat.probs.sum() == 0 or cat.probs.min() < 0:
                print(cat)
            try:
                arr.append(cat.sample())
            except Exception as e:
                print(e)
        self.last_sample = torch.stack(arr, dim=1)
        return self.last_sample.reshape(-1, self._flattened_sample_size)

    def deterministic_sample(self):
        arr = [torch.argmax(cat.probs, -1) for cat in self.cats]
        self.last_sample = torch.stack(arr, dim=1)
        return self.last_sample.reshape(-1, self._flattened_sample_size)

    def logp(self, actions):
        # # If tensor is provided, unstack it into list.
        if isinstance(actions, torch.Tensor):
            actions = torch.unbind(actions.reshape(-1, 7), dim=1)
        logps = torch.stack(
            [cat.log_prob(act) for cat, act in zip(self.cats, actions)])
        logps = logps.reshape(self._flattened_sample_size, -1)
        return torch.sum(logps, dim=0)

    def multi_entropy(self):
        return torch.stack([cat.entropy() for cat in self.cats], dim=1).reshape(-1, self._flattened_sample_size)

    def entropy(self):
        return torch.sum(self.multi_entropy(), dim=1)

    def multi_kl(self, other):
        return torch.stack(
            [
                torch.distributions.kl.kl_divergence(cat, oth_cat)
                for cat, oth_cat in zip(self.cats, other.cats)
            ],
            dim=1,
        )

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return np.sum(action_space.nvec)

ModelCatalog.register_custom_action_dist("GridnetMultiCategorical", GridnetMultiCategorical)



if __name__ == '__main__':
    # sep = os.pathsep
    # os.environ['PYTHONPATH'] = sep.join(sys.path)

    ray.init(num_gpus=1)

    env_name = 'ray-microrts-env'

    register_env(env_name, GridnetWrapper)
    ModelCatalog.register_custom_model('GridnetAgent', GridnetAgent)

    config = {
        'framework': 'torch',
        'num_workers': 1,
        'num_cpus_per_worker': 11,


        'model': {
            'custom_model': 'GridnetAgent',
            "custom_action_dist": "GridnetMultiCategorical",
            'custom_model_config': {},
        },
        'env': env_name,
        'env_config': {
            'num_bot_envs': 24,
            'max_steps': 2000,
            'render_theme': 1,
            'frame_skip': 0,
            'ai2s': [microrts_ai.passiveAI] * 24,
            'map_path': 'maps/16x16/basesWorkers16x16.xml',
            'reward_weight': np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        },

        # 'lr': tune.grid_search([0.0001, 0.0005, 0.001, 0.005])
    }

    stop = {
        'timesteps_total': 1000000,
    }

    result = tune.run(ImpalaTrainer, config=config, stop=stop)
