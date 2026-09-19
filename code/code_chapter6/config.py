"""先读这里：单位为 m / rad / s；任务目标全部写死在世界坐标系。"""
from dataclasses import dataclass

import numpy as np

from code_chapter4.kinematics import Pose2D

# 不随机生成目标，不使用视觉/导航框架/学习模型。
BASE_WAYPOINTS = (Pose2D(0.35, 0.0, 0.0), Pose2D(0.70, 0.0, 0.0))
TABLE_CENTER = (1.40, -0.42, 0.86)
TABLE_SIZE = (0.70, 0.90, 0.08)  # 台面高度 0.90
CUBE_SIZE = 0.04
CUBE_POSITION = (1.27, -0.30, 0.921)
BOX_CENTER = (1.18, -0.52)       # 开口方框中心 XY
BOX_INNER_SIZE = 0.18
BOX_WALL = 0.012
BOX_HEIGHT = 0.045
BOX_FLOOR = 0.008
TABLE_TOP = TABLE_CENTER[2] + TABLE_SIZE[2] / 2
BOX_FLOOR_TOP = TABLE_TOP + BOX_FLOOR
# TCP 的 z 轴朝下、x 轴沿世界 y：夹爪沿 y 方向夹紧。
GRASP_ROTATION = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
GRIPPER_JOINT = "idx81_gripper_r_outer_joint1"
GRIPPER_OPEN = 0.72
GRIPPER_CLOSED = 0.0


@dataclass(frozen=True)
class TaskConfig:
    mode: str = "relative_joint"
    headless: bool = False
    physics_hz: int = 120
    waypoint_timeout: float = 25.0
    arm_duration: float = 2.5
    clearance: float = 0.16
    cartesian_step: float = 0.012

    def __post_init__(self):
        if self.mode not in ("relative_joint", "delta_ee"):
            raise ValueError("mode 必须是 relative_joint 或 delta_ee")
        for name in ("physics_hz", "waypoint_timeout", "arm_duration", "clearance", "cartesian_step"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if self.physics_hz != int(self.physics_hz):
            raise ValueError("physics_hz 必须是整数")

    @property
    def dt(self):
        return 1.0 / self.physics_hz
