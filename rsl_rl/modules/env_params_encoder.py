
import numpy as np

import torch
import torch.nn as nn
# from torch.distributions import Normal
from rsl_rl.modules import get_activation

class EnvParamsEncoder(nn.Module):
    def __init__(self,  num_params_encoder,
                        latent_dim,
                        hidden_dims=[128, 128],
                        init_noise_std=1.0,
                        activation='elu',
                        **kwargs):
        if kwargs:
            print("EnvParamsEncoder.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(EnvParamsEncoder, self).__init__()

        activation = get_activation(activation)

        # Extrinsic params encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_params_encoder, hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)
        print(f"Encoder MLP: {self.encoder}")

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

    def forward(self, env_params):
        return self.encoder(env_params)
    
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