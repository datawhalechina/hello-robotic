"""固定动作脚本：准备 -> 移动 -> 预抓取 -> 下探 -> 夹紧 -> 抬升 -> 放置。

物体真值只用于验收，不用于搜索或追踪目标。全部运动目标来自 config.py。
"""
import numpy as np

from code_chapter5.config import HOME_JOINT_POSITIONS
from code_chapter5.kinematics import quaternion_to_matrix
from . import config as C


def cube_inside_box(position, quaternion):
    """整块立方体而非仅中心在框内，且位于底板上；允许 6 mm 接触容差。"""
    position = np.asarray(position)
    extent = np.abs(quaternion_to_matrix(quaternion)) @ np.full(3, C.CUBE_SIZE/2)
    inside_xy = np.all(np.abs(position[:2]-C.BOX_CENTER)+extent[:2] <= C.BOX_INNER_SIZE/2-.002)
    bottom = position[2]-extent[2]
    return bool(inside_xy and abs(bottom-C.BOX_FLOOR_TOP) < .006)


def run_task(motion):
    env, cfg = motion.env, motion.config
    motion.set_phase("prepare: 收臂、张开夹爪")
    motion.relative_joint(HOME_JOINT_POSITIONS-motion.arm.get_joint_positions())
    motion.gripper(opened=True)
    for target in C.BASE_WAYPOINTS:
        motion.navigate(target)
    parked_pose = env.base_pose()

    pick = np.array(C.CUBE_POSITION)
    pick[2] = C.TABLE_TOP+C.CUBE_SIZE/2
    above_pick = pick + [0., 0., cfg.clearance]
    place = np.array([*C.BOX_CENTER, C.BOX_FLOOR_TOP+C.CUBE_SIZE/2+.012])
    above_place = place + [0., 0., cfg.clearance]

    # 高空准备姿态统一用关节轨迹；抓放阶段由 --mode 决定。
    # 从水平伸展姿态直接做向下的全姿态直线插值，路径未必可达。
    motion.set_phase("pregrasp: 到红块上方")
    motion.move_world(above_pick, C.GRASP_ROTATION, mode="relative_joint")
    motion.set_phase("descend: 下探固定抓取点")
    motion.move_world(pick, C.GRASP_ROTATION)
    motion.set_phase("close: 夹紧")
    motion.gripper(opened=False)
    motion.set_phase("lift: 垂直抬升")
    motion.move_world(above_pick, C.GRASP_ROTATION)
    motion.hold(.4)
    cube_p, _ = env.cube.get_world_pose()
    tcp_p, _ = env.tcp_pose()
    lift_height = float(cube_p[2]-(C.TABLE_TOP+C.CUBE_SIZE/2))
    if lift_height < cfg.clearance*.6 or np.linalg.norm(cube_p-tcp_p) > .09:
        raise RuntimeError(f"抓取失败：红块仅抬升 {lift_height:.3f} m；停止搬运")
    print(f"  红块实际抬升 {lift_height:.3f} m", flush=True)

    motion.set_phase("transfer: 高空移动到框上方")
    motion.move_world(above_place, C.GRASP_ROTATION)
    motion.set_phase("place: 下放")
    motion.move_world(place, C.GRASP_ROTATION)
    motion.set_phase("release: 张开夹爪")
    motion.gripper(opened=True)
    motion.set_phase("retreat: 向上退出")
    motion.move_world(above_place, C.GRASP_ROTATION)
    motion.hold(1.0)

    # 连续半秒验收：防止方块仅短暂穿过目标区域就被标成成功。
    for _ in range(round(.5/cfg.dt)):
        p, q = env.cube.get_world_pose()
        if not cube_inside_box(p, q) or np.linalg.norm(env.cube.get_linear_velocity()) > .04:
            raise RuntimeError(f"放置失败：红块未稳定落入方框，实际位置 {p}")
        motion.step()
    final_base = env.base_pose()
    drift = np.hypot(final_base.x-parked_pose.x, final_base.y-parked_pose.y)
    if drift > .02:
        raise RuntimeError(f"抓取过程中底盘漂移过大：{drift:.3f} m")
    motion.set_phase("success")
    return {"success": True, "mode": cfg.mode, "lift_height_m": lift_height,
            "cube_position": p.tolist(), "base_drift_m": float(drift),
            "simulation_time_s": env.steps*cfg.dt}
