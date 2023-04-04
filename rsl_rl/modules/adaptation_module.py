
import numpy as np

import torch
import torch.nn as nn
# from torch.distributions import Normal
from rsl_rl.modules import get_activation

class AdaptationModule(nn.Module):
    def __init__(self,  num_obs,
                        num_actions,
                        latent_dim,
                        hidden_dims=[64, 64],
                        mlp_output_dim=32,
                        num_temporal_steps=32,
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("AdaptationModule.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(AdaptationModule, self).__init__()

        activation = get_activation(activation)

        # MLP
        mlp_layers = []
        mlp_layers.append(nn.Linear(num_obs + num_actions, hidden_dims[0]))
        mlp_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                mlp_layers.append(nn.Linear(hidden_dims[l], mlp_output_dim))
            else:
                mlp_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                mlp_layers.append(activation)
        self.mlp = nn.Sequential(*mlp_layers)
        print(f"Adaptation module MLP: {self.mlp}")

        # Temporal CNN
        self.num_temporal_steps = num_temporal_steps
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(mlp_output_dim, 32, kernel_size=8, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0))
        # Add non linearities here?
        self.linear = nn.Sequential(nn.Linear(32, latent_dim), nn.Tanh())
        print(f"Adaptation module temporal CNN: {self.temporal_cnn}")

        # # Noise. TODO?
        # self.std = nn.Parameter(init_noise_std * torch.ones(latent_dim))
        # self.distribution = None
        # # disable args validation for speedup
        # Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, state_action_history):
        # state_action_history (num_envs, num_temporal_steps, num_obs + num_actions)
        mlp_output = self.mlp(state_action_history)
        mlp_output = mlp_output.permute(0, 2, 1)
        # mlp_output (num_envs, num_temporal_steps, mlp_output_dim)
        cnn_output = self.temporal_cnn(mlp_output)
        # cnn_output (num_envs, 32, 1)
        latent = self.linear(cnn_output.flatten(1))
        # latent (num_envs, latent_dim)
        return latent
    
    # @property
    # def action_std(self):
    #     return self.distribution.stddev
    
    # @property
    # def entropy(self):
    #     return self.distribution.entropy().sum(dim=-1)

    # def update_distribution(self, states, actions):
    #     pass

    # def predict(self, states, actions, **kwargs):
    #     self.update_distribution(states, actions)
    #     return self.distribution.sample()

    # def predict_inference(self, states, actions):
    #     return self.forward(states, actions)