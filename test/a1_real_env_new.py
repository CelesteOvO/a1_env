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

class Go1MujocoEnv(gymnasium.Env):
    def __init__(self, ctrl_type="ctrl_type", **kwargs):

        self.robot = A1Robot(time_step=0.002, motor_control_mode=MotorControlMode.HYBRID, **kwargs)

        self._max_episode_time_sec = 15.0  # 最大回合时间
        self._step = 0  # 当前步数

        # 奖励权重
        self.reward_weights = {
            "linear_vel_tracking": 2.0,  # Was 1.0
            "angular_vel_tracking": 1.0,
            "healthy": 0.0,  # was 0.05
            "feet_airtime": 1.0,
        }

        # 惩罚权重
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 2.0,  # Was 1.0
            "xy_angular_vel": 0.05,  # Was 0.05
            "action_rate": 0.01,
            "joint_limit": 10.0,
            "joint_velocity": 0.01,
            "joint_acceleration": 2.5e-7,
            "orientation": 1.0,
            "collision": 1.0,
            "default_joint_position": 0.1,
        }

        self._curriculum_base = 0.3
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
        self._tracking_velocity_sigma = 0.25

        # Metrics used to determine if the episode should be terminated
        self._healthy_z_range = (0.22, 0.65)
        self._healthy_pitch_range = (-np.deg2rad(10), np.deg2rad(10))
        self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)

        # self._cfrc_ext_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL
        # self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]
        #
        # # Non-penalized degrees of freedom range of the control joints
        # dof_position_limit_multiplier = 0.9  # The % of the range that is not penalized
        #
        # ### TODO：替换self.model的内容 当前的机器人接口没有给出关节的控制范围？硬件直接给出？继续使用xml文件？
        # ctrl_range_offset = (
        #     0.5
        #     * (1 - dof_position_limit_multiplier)
        #     * (
        #         self.model.actuator_ctrlrange[:, 1]
        #         - self.model.actuator_ctrlrange[:, 0]
        #     )
        # )
        # # First value is the root joint, so we ignore it
        # self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        # self._soft_joint_range[:, 0] += ctrl_range_offset
        # self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1

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
        self._max_contact_force = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

    def step(self, action):
        self._step += 1

        # 0. 计算混合动作
        self._hybrid_action = self._compute_command(action)

        # 1. 发送动作到实际机器人
        self.robot.ApplyAction(self._hybrid_action)

        # 2. 获取机器人状态作为观测值
        self.robot.ReceiveObservation()
        observation = self._get_obs()

        # # 3. 计算奖励
        # reward, reward_info = self._calc_reward(action)
        #
        # # 4. 判断终止条件
        # terminated = not self.is_healthy
        # truncated = self._step >= (self._max_episode_time_sec / self.robot.time_step)
        #
        # # 5. 构建返回信息
        # info = {
        #     "x_position": self.robot.GetBasePosition()[0],
        #     "y_position": self.robot.GetBasePosition()[1],
        #     "distance_from_origin": np.linalg.norm(self.robot.GetBasePosition()[0:2], ord=2),
        #     **reward_info,
        # }

        self._last_action = action

        ### TODO 只有observation是有用的，reward和info是用来调试的，terminated和truncated是用来判断是否终止的
        return observation
        # return observation, reward, terminated, truncated, info

    def _compute_command(self, actions):
        # 缩放动作
        actions_scaled = actions * self.action_scale
        control_type = self.motor_control_mode
        if control_type == "T":
            torques = self.p_gains * (actions_scaled + self._default_joint_position - self.robot.GetTrueMotorAngles()) - self.d_gains * self.robot.GetMotorVelocities()
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
        ### TODO get orientation from robot interface
        # 从机器人接口获取四元数 (x, y, z, w)
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

    @property
    def feet_contact_forces(self):
        ### TODO get feet contact forces from robot interface
        feet_contact_forces = self.robot.raw_state.footForce
        return np.linalg.norm(feet_contact_forces, axis=1)

    ######### Positive Reward functions #########
    @property
    def linear_velocity_tracking_reward(self):
        ### TODO get base linear velocity from robot interface
        base_linear_velocity = self.robot.GetBaseVelocity()  # 返回形如 [vx, vy, vz] 的速度向量
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - base_linear_velocity[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def angular_velocity_tracking_reward(self):
        ### TODO get base angular velocity from robot interface
        base_angular_velocity_z = self.robot.GetBaseRollPitchYawRate()[2]  # 返回 [roll_rate, pitch_rate, yaw_rate]
        vel_sqr_error = np.square(self._desired_velocity[2] - base_angular_velocity_z)
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def feet_air_time_reward(self):
        """Award strides depending on their duration only when the feet makes contact with the ground"""
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time - 1.0) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return air_time_reward

    @property
    def healthy_reward(self):
        return self.is_healthy

    ######### Negative Reward functions #########
    @property  # TODO: Not used
    def feet_contact_forces_cost(self):
        return np.sum(
            (self.feet_contact_forces - self._max_contact_force).clip(min=0.0)
        )

    @property
    def non_flat_base_cost(self):
        # Penalize the robot for not being flat on the ground
        return np.sum(np.square(self.projected_gravity[:2]))

    @property
    def collision_cost(self):
        ### TODO cannot get collision forces from robot interface???
        # Penalize collisions on selected bodies
        return np.sum(
            1.0
            * (np.linalg.norm(self.data.cfrc_ext[self._cfrc_ext_contact_indices]) > 0.1)
        )

    @property
    def joint_limit_cost(self):
        ### TODO 直接使用xml中的关节范围进行关节计算
        # Get the current joint positions from the real robot
        current_joint_positions = self.robot.GetTrueMotorAngles()

        # Calculate the out-of-range penalty
        out_of_range = (self._soft_joint_range[:, 0] - current_joint_positions).clip(
            min=0.0
        ) + (current_joint_positions - self._soft_joint_range[:, 1]).clip(min=0.0)

        # Sum the penalties to get the total cost
        return np.sum(out_of_range)

    @property
    def torque_cost(self):
        ### TODO get motor torques from robot interface, motor.tauEst是合理的吗？
        motor_torques = np.array([motor.tauEst for motor in self.robot.raw_state.motorState[:12]])
        print("Motor Torques:", motor_torques)
        torque_cost_value = np.sum(np.square(motor_torques))
        print("Torque Cost:", torque_cost_value)
        return torque_cost_value
        # return np.sum(np.square(self.data.qfrc_actuator[-12:]))

    @property
    def vertical_velocity_cost(self):
        ### TODO get base linear velocity from robot interface
        vertical_velocity = self.robot.GetBaseVelocity()[2]
        vertical_velocity_cost_value = np.square(vertical_velocity)
        return vertical_velocity_cost_value
        #return np.square(self.data.qvel[2])

    @property
    def xy_angular_velocity_cost(self):
        ### TODO get base angular velocity from robot interface
        gyroscope_data = self.robot.GetTrueBaseRollPitchYawRate()
        xy_angular_velocity = gyroscope_data[:2]  # [wx, wy]
        xy_angular_velocity_cost_value = np.sum(np.square(xy_angular_velocity))
        return xy_angular_velocity_cost_value
        # return np.sum(np.square(self.data.qvel[3:5]))

    def action_rate_cost(self, action):
        return np.sum(np.square(self._last_action - action))

    @property
    def joint_velocity_cost(self):
        ### TODO get joint velocities from robot interface
        motor_velocities = self.robot.GetMotorVelocities()
        joint_velocity_cost_value = np.sum(np.square(motor_velocities))
        return joint_velocity_cost_value
        # return np.sum(np.square(self.data.qvel[6:]))

    @property
    def acceleration_cost(self):
        ### TODO get joint accelerations from robot interface
        motor_accelerations = np.array([motor.ddq for motor in self.robot.raw_state.motorState[:12]])
        acceleration_cost_value = np.sum(np.square(motor_accelerations))
        return acceleration_cost_value
        # return np.sum(np.square(self.data.qacc[6:]))

    @property
    def default_joint_position_cost(self):
        ### TODO get joint positions from robot interface
        motor_positions = self.robot.GetMotorAngles()
        joint_position_cost_value = np.sum(np.square(motor_positions - self._default_joint_position))
        return joint_position_cost_value
        # return np.sum(np.square(self.data.qpos[7:] - self._default_joint_position))

    @property
    def curriculum_factor(self):
        return self._curriculum_base**0.997

    def _calc_reward(self, action):
        # TODO: Add debug mode with custom Tensorboard calls for individual reward
        #   functions to get a better sense of the contribution of each reward function
        # TODO: Cost for thigh or calf contact with the ground

        # Positive Rewards
        linear_vel_tracking_reward = (
            self.linear_velocity_tracking_reward
            * self.reward_weights["linear_vel_tracking"]
        )
        angular_vel_tracking_reward = (
            self.angular_velocity_tracking_reward
            * self.reward_weights["angular_vel_tracking"]
        )
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_time_reward = (
            self.feet_air_time_reward * self.reward_weights["feet_airtime"]
        )
        rewards = (
            linear_vel_tracking_reward
            + angular_vel_tracking_reward
            + healthy_reward
            + feet_air_time_reward
        )
        #print(f"linear_vel_tracking_reward: {linear_vel_tracking_reward}, angular_vel_tracking_reward: {angular_vel_tracking_reward}, healthy_reward: {healthy_reward}, feet_air_time_reward: {feet_air_time_reward}")

        # Negative Costs
        ctrl_cost = self.torque_cost * self.cost_weights["torque"]
        action_rate_cost = (
            self.action_rate_cost(action) * self.cost_weights["action_rate"]
        )
        vertical_vel_cost = (
            self.vertical_velocity_cost * self.cost_weights["vertical_vel"]
        )
        xy_angular_vel_cost = (
            self.xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"]
        )
        joint_limit_cost = self.joint_limit_cost * self.cost_weights["joint_limit"]
        joint_velocity_cost = (
            self.joint_velocity_cost * self.cost_weights["joint_velocity"]
        )
        joint_acceleration_cost = (
            self.acceleration_cost * self.cost_weights["joint_acceleration"]
        )
        orientation_cost = self.non_flat_base_cost * self.cost_weights["orientation"]
        collision_cost = self.collision_cost * self.cost_weights["collision"]
        default_joint_position_cost = (
            self.default_joint_position_cost
            * self.cost_weights["default_joint_position"]
        )

        #print(f"ctrl_cost: {ctrl_cost}, action_rate_cost: {action_rate_cost}, vertical_vel_cost: {vertical_vel_cost}, xy_angular_vel_cost: {xy_angular_vel_cost}, joint_limit_cost: {joint_limit_cost}, joint_velocity_cost: {joint_velocity_cost}, joint_acceleration_cost: {joint_acceleration_cost}, orientation_cost: {orientation_cost}, collision_cost: {collision_cost}, default_joint_position_cost: {default_joint_position_cost}")

        costs = (
            ctrl_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
            + joint_acceleration_cost
            + orientation_cost
            + default_joint_position_cost
            # + collision_cost
        )

        reward = max(0.0, rewards - costs)
        # reward = rewards - self.curriculum_factor * costs
        reward_info = {
            "linear_vel_tracking_reward": linear_vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        #print(f"reward: {reward}, rewards: {rewards}, costs: {costs}")
        return reward, reward_info

    def reset_model(self):
        ### TODO reset
        default_motor_angles = self._default_joint_position.copy()
        reset_time = 3.0

        self.robot.ReceiveObservation()  # 获取当前状态
        current_motor_angles = self.robot.GetMotorAngles()
        for t in np.linspace(0, reset_time, int(reset_time / self.dt)):
            blend_ratio = t / reset_time
            target_angles = (1 - blend_ratio) * current_motor_angles + blend_ratio * default_motor_angles
            self.robot.ApplyAction(target_angles, motor_control_mode=MotorControlMode.POSITION)
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.dt:
                pass
            self.robot.ReceiveObservation()

        # Reset the variables and sample a new desired velocity
        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0

        observation = self._get_obs()

        return observation

    def _sample_desired_vel(self):
        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        return desired_vel

    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    # @property
    # def is_healthy(self):
    #     ### TODO change function is_healthy by using robot interface
    #     state = self.state_vector()
    #     min_z, max_z = self._healthy_z_range
    #     is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
    #
    #     min_roll, max_roll = self._healthy_roll_range
    #     is_healthy = is_healthy and min_roll <= state[4] <= max_roll
    #
    #     min_pitch, max_pitch = self._healthy_pitch_range
    #     is_healthy = is_healthy and min_pitch <= state[5] <= max_pitch
    #
    #     return is_healthy
