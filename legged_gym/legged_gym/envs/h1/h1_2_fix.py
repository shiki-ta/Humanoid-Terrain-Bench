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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1_2FixCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.05]
        default_joint_angles = {
            'left_hip_yaw_joint': 0,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.16,
            'left_knee_joint': 0.36,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,

            'right_hip_yaw_joint': 0,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.16,
            'right_knee_joint': 0.36,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,

            'torso_joint': 0,

            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0,
            'left_elbow_pitch_joint': 0.3,

            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': 0,
            'right_elbow_pitch_joint': 0.3,
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        n_scan = 132
        n_priv = 3 
        n_proprio = 46

        num_observations = n_proprio + n_scan + n_priv 
        num_actions = 12
        env_spacing = 3. 
        contact_buf_len = 100
        
    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        stiffness = {
            'hip_yaw_joint': 200.,
            'hip_roll_joint': 200.,
            'hip_pitch_joint': 200.,
            'knee_joint': 300.,
            'ankle_pitch_joint': 40.,
            'ankle_roll_joint': 40.,
        }
        damping = {
            'hip_yaw_joint': 2.5,
            'hip_roll_joint': 2.5,
            'hip_pitch_joint': 2.5,
            'knee_joint': 4,
            'ankle_pitch_joint': 2.0,
            'ankle_roll_joint': 2.0,
        } 
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_fix_arm.urdf'
        name = "h1_2_fix"
        foot_name = "ankle_roll"
        knee_name = "knee"
        hip_name = "hip_yaw"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 1 
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 0.8]
        randomize_base_mass = False
        added_mass_range = [0., 3.]
        randomize_base_com = False
        added_com_range = [-0.2, 0.2]
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.5

        randomize_motor = False
        motor_strength_range = [1., 1.] 

        delay_update_global_steps = 24 * 8000
        action_delay = False
        action_curr_step = [1, 1]
        action_curr_step_scratch = [0, 1]
        action_delay_view = 1
        action_buf_len = 8

    class commands( LeggedRobotCfg.commands ):
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.1, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0, 0]
            heading = [0, 0]

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        is_play = False
    



class H1_2FixCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'h1_2_fix'
