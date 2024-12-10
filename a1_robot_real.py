# coding=utf-8
"""Real robot interface of A1 robot."""

import os
import inspect
import sys
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from absl import logging
import math
import re
import numpy as np
import time

import transforms3d
from transforms3d.euler import quat2euler
from transforms3d.quaternions import qinverse
from transforms3d.quaternions import mat2quat
from transforms3d.affines import compose

import robot_config
from robot_interface import RobotInterface
# print(RobotInterface)

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]

# position mode PD
ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
PI = math.pi

JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
COM_OFFSET = -np.array([0.005, 0.0028, 0.000515])  # x++ back++     y++ right++
HIP_OFFSETS = np.array([[0.1805, -0.047, 0.], [0.1805, 0.047, 0.],
                        [-0.1805, -0.047, 0.], [-0.1805, 0.047, 0.]
                        ]) + COM_OFFSET


def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos(np.clip((x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) /
                                    (2 * l_low * l_up), -1.0, 1.0))
    l = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(np.clip(-x / l, -1.0, 1.0)) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])


def foot_position_in_hip_frame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = np.sqrt(l_up ** 2 + l_low ** 2 +
                           2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
    off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
    return np.array([off_x, off_y, off_z])


def analytical_leg_jacobian(leg_angles, leg_id):
    """
    Computes the analytical Jacobian.
    Args:
    ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
      l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1) ** (leg_id + 1)

    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
    l_eff = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2
    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -l_eff * np.cos(t_eff)
    J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
        t_eff) / 2
    J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
        t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
        t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
    return J


# For JIT compilation
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), 1)
foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), -1)


def foot_positions_in_base_frame(foot_angles):
    foot_angles = foot_angles.reshape((4, 3))
    foot_positions = np.zeros((4, 3))
    for i in range(4):
        foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],
                                                       l_hip_sign=(-1) ** (i + 1))
    return foot_positions + HIP_OFFSETS


class A1Robot:
    """Interface for real A1 robot."""
    # unitree a1_robot
    MPC_BODY_MASS = 10
    # MPC_BODY_INERTIA = np.array((0.0158533, 0, 0, 0, 0.0377999, 0, 0, 0, 0.0456542))  # unitree urdf
    # MPC_BODY_INERTIA = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064))  # unitree
    MPC_BODY_INERTIA = np.array((0.24, 0, 0, 0, 0.80, 0, 0, 0, 1.00))  # google xxxxxx
    MPC_BODY_HEIGHT = 0.28

    STAND_UP_HEIGHT = 0.25
    LEG_LENGTH = 0.20
    STANDUP_ABDUCTION_ANGLE = 0.0
    STANDUP_HIP_ANGLE = np.arccos(STAND_UP_HEIGHT / 2.0 / LEG_LENGTH)
    STANDUP_KNEE_ANGLE = -2.0 * STANDUP_HIP_ANGLE
    INIT_MOTOR_ANGLES = np.array([STANDUP_ABDUCTION_ANGLE, STANDUP_HIP_ANGLE, STANDUP_KNEE_ANGLE] * NUM_LEGS)

    print("Real Robot MPC_BODY_MASS: ", MPC_BODY_MASS)
    print("Real Robot MPC_BODY_INERTIA: ", MPC_BODY_INERTIA)
    print("Real Robot MPC_BODY_HEIGHT: ", MPC_BODY_HEIGHT)

    def __init__(self, time_step=0.002, **kwargs):
        """Initializes the robot class."""
        # Initialize pd gain vector
        self.motor_kps = np.array([ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] * 4)
        self.motor_kds = np.array([ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] * 4)
        self._motor_control_mode = kwargs['motor_control_mode']

        self.time_step = time_step

        # Robot state variables
        self._init_complete = False
        self._base_orientation = None
        self.raw_state = None
        self._last_raw_state = None
        self._motor_angles = np.zeros(12)
        self._motor_velocities = np.zeros(12)
        self._joint_states = None

        # Initiate UDP for robot state and actions
        self._robot_interface = RobotInterface()
        self._robot_interface.send_command(np.zeros(60, dtype=np.float32))

        self._state_action_counter = 0
        self._step_counter = 0
        self.estimated_velocity = np.zeros(3)
        self._last_reset_time = time.time()
        self._init_complete = True

    def ReceiveObservation(self):
        """Receives observation from robot.
        """
        state = self._robot_interface.receive_observation()
        self.raw_state = state
        q = state.imu.quaternion
        # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
        self._base_orientation = np.array([q[1], q[2], q[3], q[0]])  # xyzw
        # self._base_orientation = np.array([0, 0, 0, 1])  # xyzw
        self._motor_angles = np.array([motor.q for motor in state.motorState[:12]])
        self._motor_velocities = np.array(
            [motor.dq for motor in state.motorState[:12]])
        self._joint_states = np.array(
            list(zip(self._motor_angles, self._motor_velocities)))

    def GetTrueMotorAngles(self):
        return self._motor_angles.copy()

    def GetMotorAngles(self):
        return self._motor_angles.copy()

    def GetMotorVelocities(self):
        return self._motor_velocities.copy()

    def GetBasePosition(self):
        return -1

    def GetBaseRollPitchYaw(self):
        return self.getEulerFromQuaternion(self._base_orientation)

    def GetTrueBaseRollPitchYaw(self):
        return self.getEulerFromQuaternion(self._base_orientation)

    def GetBaseRollPitchYawRate(self):
        return self.GetTrueBaseRollPitchYawRate()

    def GetTrueBaseRollPitchYawRate(self):
        return np.array(self.raw_state.imu.gyroscope).copy()

    def GetBaseVelocity(self):
        return self.estimated_velocity.copy()

    def GetFootContacts(self):
        return np.array(self.raw_state.footForce) > 20

    def GetTimeSinceReset(self):
        return time.time() - self._last_reset_time

    def GetBaseOrientation(self):
        return self._base_orientation.copy()

    def GetTrueBaseOrientation(self):
        return self._base_orientation.copy()

    @property
    def motor_velocities(self):
        return self._motor_velocities.copy()

    def ApplyAction(self, motor_commands, motor_control_mode=None):
        """Clips and then apply the motor commands using the motor model.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).
          motor_control_mode: A MotorControlMode enum.
        """
        if motor_control_mode is None:
            motor_control_mode = self._motor_control_mode

        command = np.zeros(60, dtype=np.float32)
        if motor_control_mode == robot_config.MotorControlMode.POSITION:
            for motor_id in range(NUM_MOTORS):
                command[motor_id * 5] = motor_commands[motor_id]
                command[motor_id * 5 + 1] = self.motor_kps[motor_id]
                command[motor_id * 5 + 3] = self.motor_kds[motor_id]
        elif motor_control_mode == robot_config.MotorControlMode.TORQUE:
            for motor_id in range(NUM_MOTORS):
                command[motor_id * 5 + 4] = motor_commands[motor_id]
        elif motor_control_mode == robot_config.MotorControlMode.HYBRID:
            command = np.array(motor_commands, dtype=np.float32)
        else:
            raise ValueError('Unknown motor control mode for A1 robot: {}.'.format(
                motor_control_mode))

        self._robot_interface.send_command(command)

    def Reset(self, default_motor_angles=None, reset_time=3.0):
        """Reset the robot to default motor angles."""
        # self._velocity_estimator.reset()
        self._state_action_counter = 0
        self._step_counter = 0
        self._last_reset_time = time.time()

    def Step(self, action, motor_control_mode=None):

        # apply_action_start = time.time()
        self.ApplyAction(action, motor_control_mode)
        # print("[StepInternal] ApplyAction time: ", 1000 * (time.time() - apply_action_start), " ms")

        # observation_start = time.time()
        self.ReceiveObservation()
        # print("[StepInternal] ReceiveObservation time: ", 1000 * (time.time() - observation_start), " ms")
        self._state_action_counter += 1

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                                foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local position.

        Args:
          leg_id: The leg index.
          foot_local_position: The foot link's position in the base frame.

        Returns:
          A tuple. The position indices and the angles for all joints along the
          leg. The position indices is consistent with the joint orders as returned
          by GetMotorAngles API.
        """
        assert len(self._foot_link_ids) == self.num_legs
        # toe_id = self._foot_link_ids[leg_id]

        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = list(
            range(leg_id * motors_per_leg,
                  leg_id * motors_per_leg + motors_per_leg))

        joint_angles = foot_position_in_hip_frame_to_joint_angle(
            foot_local_position - HIP_OFFSETS[leg_id],
            l_hip_sign=(-1) ** (leg_id + 1))

        # Joint offset is necessary for Laikago.
        joint_angles = np.multiply(
            np.asarray(joint_angles) -
            np.asarray(self._motor_offset)[joint_position_idxs],
            self._motor_direction[joint_position_idxs])

        # Return the joing index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles.tolist()

    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        motor_angles = self.GetMotorAngles()
        return foot_positions_in_base_frame(motor_angles)

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Does not work for Minitaur which has the four bar mechanism for now.
        motor_angles = self.GetMotorAngles()[leg_id * 3:(leg_id + 1) * 3]
        return analytical_leg_jacobian(motor_angles, leg_id)

    def getEulerFromQuaternion(self, q):
        q = self.quat2wxyz(q)  # 如果这一步是自定义的，保留
        return quat2euler(q, axes='sxyz')  # 替换为 transforms3d 的方法

    def invertTransform(self, t, q):
        q = self.quat2wxyz(q)
        try:
            q = qinverse(q)  # 替换为 transforms3d 的方法
        except ValueError:
            print("ValueError: not a valid quaternion")
        q = self.quat2xyzw(q)
        return (-t[0], -t[1], -t[2]), q

    # def multiplyTransforms(self, t1, q1, t2, q2):
    #     q1 = self.quat2wxyz(q1)
    #     q2 = self.quat2wxyz(q2)
    #     T1 = transformations.quaternion_matrix(q1)
    #     T2 = transformations.quaternion_matrix(q2)
    #     T1[:3, 3] = t1
    #     T2[:3, 3] = t2
    #     T = T1.dot(T2)
    #     p = T[:3, 3]
    #     q = transformations.quaternion_from_matrix(T)
    #     q = self.quat2xyzw(q)
    #     return p, q

    def multiplyTransforms(self, t1, q1, t2, q2):
        q1 = self.quat2wxyz(q1)
        q2 = self.quat2wxyz(q2)
        T1 = compose(t1, q1, [1, 1, 1])  # 平移 t1, 旋转 q1, 缩放为单位缩放
        T2 = compose(t2, q2, [1, 1, 1])
        T = T1.dot(T2)
        p = T[:3, 3]
        q = mat2quat(T[:3, :3])
        q = self.quat2xyzw(q)
        return p, q

    # def getMatrixFromQuaternion(self, q):
    #     q = self.quat2wxyz(q)
    #     return transformations.quaternion_matrix(q)[:3, :3]

    def getMatrixFromQuaternion(self, q):
        q = self.quat2wxyz(q)
        return compose([0, 0, 0], q, [1, 1, 1])[:3, :3]  # 只提取旋转部分

    def quat2wxyz(self, q):
        return (q[3], q[0], q[1], q[2])

    def quat2xyzw(self, q):
        return (q[1], q[2], q[3], q[0])

    def stand_up(self, standup_time=1.5, reset_time=5, default_motor_angles=None):
        logging.warning(
            "About to reset the robot, make sure the robot is hang-up.")

        if not default_motor_angles:
            default_motor_angles = self.INIT_MOTOR_ANGLES

        self.ReceiveObservation()
        current_motor_angles = self.GetMotorAngles()
        # Stand up in 1.5 seconds, and keep the behavior in this way.
        # standup_time = min(reset_time, 1.5)
        tik = time.time()
        count = 0
        force_list = []
        for t in np.arange(0, reset_time, self.time_step):
            count += 1
            stand_up_last_t = time.time()
            blend_ratio = min(t / standup_time, 1)
            action = blend_ratio * default_motor_angles + (
                    1 - blend_ratio) * current_motor_angles
            self.Step(action, robot_config.MotorControlMode.POSITION)
            force_list.append(self.raw_state.footForce)
            while time.time() - stand_up_last_t < self.time_step:
                pass
        tok = time.time()
        print("stand up cost time =  ", -tik + tok)
        print("count:", count)

def main():
    # 初始化机器人接口
    robot = A1Robot(motor_control_mode=robot_config.MotorControlMode.POSITION)

    # 获取当前状态
    robot.ReceiveObservation()

    # 打印每个 motor 的所有属性
    print("Motor States (Attributes):")
    for i, motor in enumerate(robot.raw_state.motorState[:12]):
        print(f"\nMotor {i} ({MOTOR_NAMES[i]}):")
        # 使用 dir() 列出所有属性
        attributes = dir(motor)
        for attr in attributes:
            # 排除私有方法和属性
            if not attr.startswith("_"):
                try:
                    value = getattr(motor, attr)
                    print(f"  {attr}: {value}")
                except Exception as e:
                    print(f"  {attr}: Error retrieving value ({e})")

    print("\nAdditional Information:")
    print("Base Orientation (quaternion):", robot.GetBaseOrientation())
    print("Base Roll Pitch Yaw:", robot.GetBaseRollPitchYaw())
    print("Estimated Velocity:", robot.GetBaseVelocity())
    print("Foot Contacts:", robot.GetFootContacts())

if __name__ == "__main__":
    main()