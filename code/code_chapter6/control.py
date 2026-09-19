"""动作接口。底盘来自第四章，关节轨迹与 IK 来自第五章。

relative_joint(dq): dq 为 7 个关节角增量。
delta_ee(dp, dr): dp 与旋转向量 dr 均表达在 arm_base_link 坐标系，
                 不是工具自身坐标系；旋转左乘当前姿态。
"""
import math

import numpy as np

from code_chapter4.base_controller import G2BaseController, PoseController
from code_chapter4.config import ControlLimits, RobotGeometry
from code_chapter4.kinematics import SwerveKinematics
from code_chapter5.arm_controller import G2ArmController
from code_chapter5.kinematics import axis_rotation, orientation_error
from .config import GRIPPER_JOINT, GRIPPER_OPEN, GRIPPER_CLOSED


def vector(values, size, name):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是 {size} 个有限数")
    return result


def rotation_increment(rotvec):
    dr = vector(rotvec, 3, "dr")
    angle = float(np.linalg.norm(dr))
    return np.eye(3) if angle < 1e-12 else axis_rotation(dr/angle, angle)[:3, :3]


class MotionController:
    def __init__(self, env, recorder=None):
        self.env, self.config, self.recorder = env, env.config, recorder
        geometry = RobotGeometry()
        self.base = G2BaseController(env.robot, SwerveKinematics(
            geometry.wheel_positions, geometry.wheel_radius), ControlLimits())
        self.pose_controller = PoseController(max_linear_speed=.22, max_angular_speed=.3,
                                             position_tolerance=.006, yaw_tolerance=.012)
        self.arm = G2ArmController(env.robot, "right")
        self.gripper_index = env.robot.dof_names.index(GRIPPER_JOINT)
        self.phase = "initialize"
        self.target_q = self.arm.get_joint_positions()
        self.gripper_target = GRIPPER_OPEN

    def set_phase(self, phase):
        self.phase = phase
        print(f"[第六章] {phase}", flush=True)

    def step(self):
        self.env.step()
        if self.recorder:
            self.recorder.sample(self)

    def hold(self, seconds):
        for _ in range(math.ceil(seconds/self.config.dt)):
            self.step()

    def navigate(self, target):
        self.set_phase(f"navigate({target.x:.2f}, {target.y:.2f}, {target.yaw:.2f})")
        try:
            for _ in range(math.ceil(self.config.waypoint_timeout/self.config.dt)):
                result = self.pose_controller.compute(self.env.base_pose(), target)
                if result.reached:
                    self.base.stop()
                    self.hold(.5)
                    # 停车惯性后再检查，不能刚经过容差区就认为成功。
                    if self.pose_controller.compute(self.env.base_pose(), target).reached:
                        return
                command = result.command
                self.base.set_velocity(command.vx, command.vy, command.wz)
                self.base.update(self.config.dt)
                self.step()
            raise RuntimeError(f"固定航点超时：{target}；实际 {self.env.base_pose()}")
        finally:
            self.base.stop()

    def relative_joint(self, delta, duration=None):
        """q_goal = q_measured + dq，再由第五章三次时间缩放执行。"""
        dq = vector(delta, 7, "dq")
        goal = self.arm.get_joint_positions() + dq
        self._check_joint_goal(goal)
        duration = self.config.arm_duration if duration is None else duration
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration 必须为有限正数")
        trajectory = self.arm.make_trajectory(goal, duration, self.config.dt)
        for target in trajectory:
            self.target_q = target.copy()
            self.arm.command_positions(target)
            self.step()
        self.hold(.3)
        self._check_tracking(goal)
        return self.arm.get_joint_positions()

    def _check_joint_goal(self, goal):
        if np.any(goal < self.arm.lower_limits-1e-8) or np.any(goal > self.arm.upper_limits+1e-8):
            raise ValueError("关节目标越界；不静默裁剪成另一个动作")

    def solve(self, position, rotation, seed=None):
        result = self.arm.solve_ik(position, rotation, seed)
        if not result.success:
            raise RuntimeError(f"第五章 IK 未收敛：位置误差 {result.position_error:.4f} m，"
                               f"姿态误差 {result.orientation_error:.4f} rad")
        return result.joint_positions

    def delta_ee(self, dp, dr=(0., 0., 0.), duration=None):
        """直线位置插值 + 旋转向量插值；预检整段 IK 后才下发关节轨迹。"""
        dp, dr = vector(dp, 3, "dp"), vector(dr, 3, "dr")
        start = self.arm.forward_kinematics()
        count = max(1, math.ceil(np.linalg.norm(dp)/self.config.cartesian_step),
                    math.ceil(np.linalg.norm(dr)/.08))
        q = self.arm.get_joint_positions()
        goals = []
        for alpha in np.linspace(0., 1., count+1)[1:]:
            q = self.solve(start.position+alpha*dp,
                           rotation_increment(alpha*dr) @ start.rotation, q)
            self._check_joint_goal(q)
            goals.append(q)
        duration = self.config.arm_duration if duration is None else duration
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration 必须为有限正数")
        # 相邻小段共用第五章轨迹生成器，不调用外部笛卡尔控制器。
        for q in goals:
            trajectory = self.arm.make_trajectory(q, duration/count, self.config.dt)
            for target in trajectory:
                self.target_q = target.copy()
                self.arm.command_positions(target)
                self.step()
        self.hold(.3)
        self._check_tracking(goals[-1])
        return self.arm.forward_kinematics()

    def move_world(self, position, rotation, mode=None):
        """方便任务调用的适配层：固定世界目标 -> 两种增量动作。"""
        position = vector(position, 3, "world_position")
        p, r = self.env.world_to_arm(position, rotation)
        mode = self.config.mode if mode is None else mode
        if mode == "relative_joint":
            q = self.solve(p, r)
            self.relative_joint(q-self.arm.get_joint_positions())
        elif mode == "delta_ee":
            current = self.arm.forward_kinematics()
            self.delta_ee(p-current.position, orientation_error(r, current.rotation))
        else:
            raise ValueError("未知动作模式")
        actual_p, actual_r = self.env.tcp_pose()
        error = float(np.linalg.norm(actual_p-position))
        angle = float(np.linalg.norm(orientation_error(rotation, actual_r)))
        print(f"  TCP 误差 {error*1000:.1f} mm / {angle:.3f} rad", flush=True)
        if error > .012 or angle > .06:
            raise RuntimeError("实测 TCP 未到达；检查坐标系、底盘漂移或关节跟踪")

    def _check_tracking(self, goal):
        error = float(np.max(np.abs(self.arm.get_joint_positions()-goal)))
        if not np.isfinite(error) or error > .06:
            raise RuntimeError(f"关节跟踪误差过大：{error:.3f} rad")

    def gripper(self, opened):
        """只驱动右夹爪主动关节；原 USD mimic/闭环连杆带动其余关节。"""
        from isaacsim.core.utils.types import ArticulationAction
        self.gripper_target = GRIPPER_OPEN if opened else GRIPPER_CLOSED
        self.env.robot.apply_action(ArticulationAction(
            joint_positions=np.array([self.gripper_target]),
            joint_indices=np.array([self.gripper_index])))
        self.hold(1.0)

    def stop(self):
        self.base.stop()
