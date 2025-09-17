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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args,  task_registry
from terrain_base.config import terrain_config

import torch
import faulthandler

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    faulthandler.enable()
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 100
    env_cfg.commands.resampling_time = 60
    env_cfg.rewards.is_play = True

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.num_goals = 10
    env_cfg.terrain.height = [0.01, 0.02]

    # prepare environment
    env: HumanoidRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    file_path = '/home/gtx/competition_only/legged_gym/logs/parkour_new/test/traced/test-0-base_jit.pt'
    policy = torch.jit.load(file_path, map_location=env.device)
    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    for i in range(10*int(env.max_episode_length)):
        depth_latent = None
        obs = obs[:,:178]
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        """
        evaluate code 
        """
        
    print("Evaluate finished!")
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
