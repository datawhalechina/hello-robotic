"""可选的数据采集：以约 20 Hz 流式写 JSONL，不存图像、不引入学习依赖。"""
import json
from pathlib import Path


class TrajectoryRecorder:
    def __init__(self, path, config):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", encoding="utf-8")  # 不覆盖已有采集结果
        self.interval = max(1, round(config.physics_hz/20))
        self.write({"type": "metadata", "schema_version": 1, "mode": config.mode,
                    "physics_dt": config.dt, "sample_interval_steps": self.interval,
                    "units": "meter, radian, second", "quaternion_order": "wxyz"})

    def write(self, row):
        self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        self.file.flush()

    def sample(self, motion):
        env = motion.env
        if env.steps % self.interval:
            return
        base = env.base_pose()
        p, q = env.cube.get_world_pose()
        tcp, _ = env.tcp_pose()
        v = motion.base.applied_velocity
        self.write({"type": "sample", "time": env.steps*env.config.dt, "phase": motion.phase,
                    "base_pose": [base.x, base.y, base.yaw], "base_command": [v.vx, v.vy, v.wz],
                    "arm_position": motion.arm.get_joint_positions().tolist(),
                    "arm_target": motion.target_q.tolist(), "gripper_target": motion.gripper_target,
                    "gripper_position": float(env.robot.get_joint_positions()[motion.gripper_index]),
                    "tcp_world": tcp.tolist(), "cube_world": p.tolist(), "cube_quaternion": q.tolist()})

    def close(self):
        self.file.close()
