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
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, MLPPolicy
from rsl_rl.storage import BCRolloutStorage

class BehaviorCloning:
    teacher: ActorCritic
    student: MLPPolicy
    def __init__(self,
                 teacher,
                 student,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 learning_rate=1e-3,
                 schedule="fixed",
                 device='cpu',
                 **kwargs):
        if kwargs:
            print("BehaviorCloning.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        self.device = device

        self.schedule = schedule
        self.learning_rate = learning_rate

        # BC components
        self.teacher = teacher
        self.student = student
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.student.parameters(), lr=learning_rate)
        self.transition = BCRolloutStorage.Transition()

        # Learning parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.teacher.eval()

    def init_storage(self, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, action_shape):
        self.storage = BCRolloutStorage(num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.student.eval()
    
    def train_mode(self):
        self.student.train()

    def act(self, student_obs, teacher_obs):
        # Compute the actions and values
        self.transition.student_actions = self.student.act(student_obs).detach()
        self.transition.teacher_actions = self.teacher.act(teacher_obs).detach()
        # need to record obs and critic_obs before env.step()
        self.transition.student_observations = student_obs
        self.transition.teacher_observations = teacher_obs
        return self.transition.teacher_actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.teacher.reset(dones)
        self.student.reset(dones)

    def update(self):
        mse_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for student_obs_batch, teacher_obs_batch, actions_batch, _, _ in generator:
            
            actions_student = self.student.act_inference(student_obs_batch)
            actions_teacher = actions_batch

            # Loss
            loss = nn.MSELoss()(actions_student, actions_teacher)
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mse_loss += loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mse_loss /= num_updates
        self.storage.clear()

        return mse_loss