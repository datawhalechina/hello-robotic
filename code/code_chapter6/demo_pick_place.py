"""第六章唯一运行入口：固定航点导航与真实接触抓放（无模型推理）。"""
import argparse
import json
from pathlib import Path
import sys

# 只增加兄弟章节的父目录；不让多个章节的 config.py 发生同名导入冲突。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_chapter6.config import TaskConfig
from code_chapter6.control import MotionController
from code_chapter6.environment import PickPlaceEnvironment
from code_chapter6.recording import TrajectoryRecorder
from code_chapter6.task import run_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("relative_joint", "delta_ee"), default="relative_joint")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true", help="成功后保留窗口观察")
    parser.add_argument("--record", type=Path, help="可选 JSONL 轨迹路径，已存在则拒绝覆盖")
    args = parser.parse_args()
    if args.headless and args.keep_open:
        parser.error("--keep-open 仅适用于有窗口模式")
    cfg = TaskConfig(mode=args.mode, headless=args.headless)
    env = motion = recorder = None
    try:
        # 先检查输出路径，再启动耗时的仿真。
        if args.record:
            recorder = TrajectoryRecorder(args.record, cfg)
        env = PickPlaceEnvironment(cfg)
        motion = MotionController(env, recorder)
        result = run_task(motion)
        print("[结果] "+json.dumps(result, ensure_ascii=False), flush=True)
        if recorder:
            recorder.write({"type": "result", **result})
        if args.keep_open:
            while env.app.is_running() and env.world.is_playing():
                env.step()
    except Exception as exc:
        if recorder:
            recorder.write({"type": "result", "success": False, "error": str(exc),
                            "phase": motion.phase if motion else "initialization"})
        raise  # 失败返回非零退出码，不能伪装成成功
    finally:
        try:
            if motion:
                motion.stop()
        finally:
            if recorder:
                recorder.close()
            if env:
                env.close()


if __name__ == "__main__":
    main()
