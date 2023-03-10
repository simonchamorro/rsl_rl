# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class BCRolloutStorage:
    class Transition:
        def __init__(self):
            self.student_observations = None
            self.teacher_observations = None
            self.student_actions = None
            self.teacher_actions = None
            self.rewards = None
            self.dones = None
            self.hidden_states = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape, device='cpu', num_epochs_stored=1):
        # TODO add support for keeping older epochs data

        self.device = device

        self.student_obs_shape = student_obs_shape
        self.teacher_obs_shape = teacher_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.student_observations = torch.zeros(num_epochs_stored, num_transitions_per_env, num_envs, *student_obs_shape, device=self.device)
        self.teacher_observations = torch.zeros(num_epochs_stored, num_transitions_per_env, num_envs, *teacher_obs_shape, device=self.device)
        self.rewards = torch.zeros(num_epochs_stored, num_transitions_per_env, num_envs, 1, device=self.device)
        self.student_actions = torch.zeros(num_epochs_stored, num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.teacher_actions = torch.zeros(num_epochs_stored, num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_epochs_stored, num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0
        self.epoch = 0
        self.num_epochs_stored = num_epochs_stored

    def add_transitions(self, transition: Transition):
        epoch = self.epoch % self.num_epochs_stored
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.student_observations[epoch, self.step].copy_(transition.student_observations)
        self.teacher_observations[epoch, self.step].copy_(transition.teacher_observations)
        self.student_actions[epoch, self.step].copy_(transition.student_actions)
        self.teacher_actions[epoch, self.step].copy_(transition.teacher_actions)
        self.rewards[epoch, self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[epoch, self.step].copy_(transition.dones.view(-1, 1))
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        # TODO
        if hidden_states is None or hidden_states==(None, None):
            return
        else:
            raise NotImplementedError
        # # make a tuple out of GRU hidden state sto match the LSTM format
        # hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        # hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # # initialize if needed 
        # if self.saved_hidden_states_a is None:
        #     self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
        #     self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # # copy the states
        # for i in range(len(hid_a)):
        #     self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
        #     self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0
        self.epoch += 1

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        num_epochs_stored = min(self.num_epochs_stored, self.epoch + 1)
        num_mini_batches = num_mini_batches * num_epochs_stored
        batch_size = self.num_envs * self.num_transitions_per_env * num_epochs_stored
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)
        student_observations = self.student_observations[:num_epochs_stored].flatten(0, 2)
        teacher_observations = self.teacher_observations[:num_epochs_stored].flatten(0, 2)

        student_actions = self.student_actions[:num_epochs_stored].flatten(0, 2)
        teacher_actions = self.student_actions[:num_epochs_stored].flatten(0, 2)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                student_obs_batch = student_observations[batch_idx]
                teacher_obs_batch = teacher_observations[batch_idx]
                student_actions_batch = student_actions[batch_idx]
                teacher_actions_batch = teacher_actions[batch_idx]
                yield student_obs_batch, teacher_obs_batch, student_actions_batch, teacher_actions_batch, (None, None), None

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # TODO
        raise NotImplementedError
        # padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.student_observations, self.dones)
        # padded_teacher_obs_trajectories, _ = split_and_pad_trajectories(self.teacher_observations, self.dones)

        # mini_batch_size = self.num_envs // num_mini_batches
        # for ep in range(num_epochs):
        #     first_traj = 0
        #     for i in range(num_mini_batches):
        #         start = i*mini_batch_size
        #         stop = (i+1)*mini_batch_size

        #         dones = self.dones.squeeze(-1)
        #         last_was_done = torch.zeros_like(dones, dtype=torch.bool)
        #         last_was_done[1:] = dones[:-1]
        #         last_was_done[0] = True
        #         trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
        #         last_traj = first_traj + trajectories_batch_size
                
        #         masks_batch = trajectory_masks[:, first_traj:last_traj]
        #         student_obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
        #         teacher_obs_batch = padded_teacher_obs_trajectories[:, first_traj:last_traj]
        #         student_actions_batch = self.student_actions[:, start:stop]
        #         teacher_actions_batch = self.teacher_actions[:, start:stop]

        #         # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
        #         # then take only time steps after dones (flattens num envs and time dimensions),
        #         # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
        #         last_was_done = last_was_done.permute(1, 0)
        #         hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
        #                         for saved_hidden_states in self.saved_hidden_states_a ] 
        #         hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
        #                         for saved_hidden_states in self.saved_hidden_states_c ]
        #         # remove the tuple for GRU
        #         hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
        #         hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

        #         yield student_obs_batch, teacher_obs_batch, student_actions_batch, teacher_actions_batch, (hid_a_batch, hid_c_batch), masks_batch
                
        #         first_traj = last_traj