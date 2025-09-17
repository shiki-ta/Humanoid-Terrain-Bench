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

class G1FixCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.78]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,
           'left_hip_roll_joint' : 0,
           'left_hip_pitch_joint' : -0.1,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_hip_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
           'torso_joint' : 0.
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
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
        name = "g1_fix_upper"
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.2, 0.6]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values

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

        only_positive_rewards = True 
        tracking_sigma = 0.25 
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.

class G1FixCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticRecurrent'
        run_name = ''
        experiment_name = 'g1_fix'
        max_iterations = 50001 # number of policy updates
        save_interval = 300
