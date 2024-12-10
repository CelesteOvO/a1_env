import gymnasium
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from scipy.spatial.transform import Rotation as R

import mujoco
import time

import numpy as np
from pathlib import Path
from a1_robot_real import A1Robot
from robot_config import MotorControlMode


class A1RealEnv(gymnasium.Env):
    def __init__(self, **kwargs):
        self.robot = A1Robot(time_step=0.002, motor_control_mode=MotorControlMode.HYBRID, **kwargs)
        self._gravity_vector = np.array([0.0, 0.0, -9.81])

        ### TODO: change default joint position
        self._default_joint_position = self.robot.INIT_MOTOR_ANGLES

        # vx (m/s), vy (m/s), wz (rad/s)
        ### TODO: 修改速度设置，是否需要从外部接口传入?
        self._desired_velocity_min = np.array([0.5, -0.0, -0.0])
        self._desired_velocity_max = np.array([0.5, 0.0, 0.0])
        self._desired_velocity = self._sample_desired_vel()  # [0.5, 0.0, 0.0]
        # self._desired_velocity = np.array([0.5, 0.0, 0.0])

        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        self.commands_scale = np.array([
            self._obs_scale["linear_velocity"],
            self._obs_scale["linear_velocity"],
            self._obs_scale["angular_velocity"]
        ])

        action_dim = 12
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.action_scale = 0.25
        self.motor_control_mode = "P"  # "P" for position control, "T" for torque control
        self.p_gains = 20.0
        self.d_gains = 0.5
        self.torque_limits = np.array([20, 55, 55, 20, 55, 55, 20, 55, 55, 20, 55, 55])

        # Action: 12 torque values
        self._hybrid_action = np.zeros(60, dtype=np.float32)
        self._last_action = np.zeros(12)
        self._clip_obs_threshold = 100.0

    def step(self, action):
        # 0. 计算混合动作
        self._hybrid_action = self._compute_command(action)

        # 1. 发送动作到实际机器人
        self.robot.ApplyAction(self._hybrid_action)
        start_time = time.time()
        while time.time() - start_time < self.robot.time_step:
            pass

        # 2. 获取机器人状态作为观测值
        self.robot.ReceiveObservation()
        observation = self._get_obs()

        self._last_action = action

        return observation

    def _compute_command(self, actions):
        actions_scaled = actions * self.action_scale
        control_type = self.motor_control_mode
        if control_type == "T":
            torques = self.p_gains * (
                        actions_scaled + self._default_joint_position - self.robot.GetTrueMotorAngles()) - self.d_gains * self.robot.GetMotorVelocities()
            torques = np.clip(torques, -self.torque_limits, self.torque_limits)
            for index in range(len(torques)):
                self._hybrid_action[5 * index + 4] = torques[index]
        elif control_type == "P":
            positions = (actions_scaled + self._default_joint_position).detach().cpu().numpy()
            for index in range(len(positions)):
                self._hybrid_action[5 * index + 0] = positions[index]  # 位置
                self._hybrid_action[5 * index + 1] = self.p_gains  # 比例增益
                self._hybrid_action[5 * index + 2] = 0  # 速度（不需要设置）
                self._hybrid_action[5 * index + 3] = self.d_gains  # 阻尼
                self._hybrid_action[5 * index + 4] = 0  # 扭矩（不需要设置）
        return self._hybrid_action

    @property
    def projected_gravity(self):
        q = self.robot.GetBaseOrientation()
        quaternion = [q[0], q[1], q[2], q[3]]  # 四元数 (x, y, z, w)

        # 从四元数转换为旋转矩阵
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # 计算局部坐标系下的重力方向
        gravity_world = self._gravity_vector  # 世界坐标系下的重力矢量
        gravity_local = rotation_matrix.T @ gravity_world  # 将重力矢量变换到局部坐标系
        gravity_local_normalized = gravity_local / np.linalg.norm(gravity_local)
        return gravity_local_normalized

    def _get_obs(self):
        dof_pos = self.robot.GetTrueMotorAngles() - self._default_joint_position
        dof_vel = self.robot.GetMotorVelocities()

        base_linear_velocity = self.robot.GetBaseVelocity()  # 线速度
        base_angular_velocity = self.robot.GetTrueBaseRollPitchYawRate()  # 角速度

        desired_vel = self._desired_velocity
        last_action = self._last_action

        projected_gravity = self.projected_gravity

        # 修改 desired_vel 的缩放方式，使用 commands_scale
        scaled_desired_vel = desired_vel * self.commands_scale

        curr_obs = np.concatenate(
            (
                base_linear_velocity * self._obs_scale["linear_velocity"],
                base_angular_velocity * self._obs_scale["angular_velocity"],
                projected_gravity,
                desired_vel * self._obs_scale["linear_velocity"],
                # scaled_desired_vel,
                dof_pos * self._obs_scale["dofs_position"],
                dof_vel * self._obs_scale["dofs_velocity"],
                last_action,
            )
        ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)

        return curr_obs

    def _sample_desired_vel(self):
        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        return desired_vel