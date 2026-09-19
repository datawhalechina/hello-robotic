"""独立场景：原始 G2 + 地面 + 台面 + 红块 + 五块长方体组成的开口框。

不导入第九章，也不修改共享 USD。无吸附、焊接、瞬移或关闭方块重力。
"""
import math

import numpy as np

from code_chapter4.config import ROBOT_USD, ROBOT_PRIM_PATH
from code_chapter4.kinematics import Pose2D
from code_chapter5.kinematics import quaternion_to_matrix
from . import config as C


class PickPlaceEnvironment:
    def __init__(self, config):
        self.config = config
        self.app = None
        self.steps = 0
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(ROBOT_USD)
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": config.headless,
                                  "disable_viewport_updates": config.headless,
                                  "renderer": "RaytracedLighting", "limit_cpu_threads": 16})
        try:
            self._build()
        except BaseException:
            self.close()
            raise

    def _build(self):
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdLux, UsdPhysics, UsdShade, PhysxSchema

        self.world = World(stage_units_in_meters=1.0, physics_dt=self.config.dt,
                           rendering_dt=1 / 60)
        self.world.scene.add_default_ground_plane()
        light = UsdLux.DomeLight.Define(self.world.stage, "/World/Light")
        light.CreateIntensityAttr(800.0)
        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        material = PhysicsMaterial("/World/Chapter6/ContactMaterial", static_friction=1.2,
                                   dynamic_friction=1.0, restitution=0.0)
        # 原模型手指已有碰撞网格；只在本次场景设置接触材料和求解精度。
        for prim in self.world.stage.Traverse():
            if str(prim.GetPath()).startswith("/genie/gripper_r") and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material.material, UsdShade.Tokens.weakerThanDescendants, "physics")
        root = self.world.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        # ArticulationRootAPI 可能位于子节点，逐一查找，不硬编码根关节。
        for prim in self.world.stage.Traverse():
            if prim.GetPath().HasPrefix(root.GetPath()) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateSolverPositionIterationCountAttr(32)
                api.CreateSolverVelocityIterationCountAttr(4)

        def fixed(name, position, size, color):
            return self.world.scene.add(FixedCuboid(
                f"/World/Chapter6/{name}", name=name, position=np.array(position),
                scale=np.array(size), size=1.0, color=np.array(color), physics_material=material))

        fixed("Table", C.TABLE_CENTER, C.TABLE_SIZE, (0.48, 0.51, 0.55))
        # 四条桌腿仅为静态几何，不增加控制逻辑。
        leg_height = C.TABLE_CENTER[2]-C.TABLE_SIZE[2]/2
        for i, (dx, dy) in enumerate(((-.28, -.36), (-.28, .36), (.28, -.36), (.28, .36))):
            fixed(f"Leg{i}", (C.TABLE_CENTER[0]+dx, C.TABLE_CENTER[1]+dy, leg_height/2),
                  (.045, .045, leg_height), (.30, .32, .35))
        x, y = C.BOX_CENTER
        width, wall = C.BOX_INNER_SIZE, C.BOX_WALL
        fixed("BoxFloor", (x, y, C.TABLE_TOP+C.BOX_FLOOR/2),
              (width+2*wall, width+2*wall, C.BOX_FLOOR), (.12, .35, .65))
        z = C.BOX_FLOOR_TOP+C.BOX_HEIGHT/2
        for i, sign in enumerate((-1, 1)):
            fixed(f"BoxX{i}", (x+sign*(width+wall)/2, y, z),
                  (wall, width+2*wall, C.BOX_HEIGHT), (.12, .35, .65))
            fixed(f"BoxY{i}", (x, y+sign*(width+wall)/2, z),
                  (width, wall, C.BOX_HEIGHT), (.12, .35, .65))
        self.cube = self.world.scene.add(DynamicCuboid(
            "/World/Chapter6/RedCube", name="red_cube", position=np.array(C.CUBE_POSITION),
            size=C.CUBE_SIZE, mass=0.04, color=np.array([.9, .025, .025]), physics_material=material))
        self.world.play()
        self.robot = self.world.scene.add(SingleArticulation(ROBOT_PRIM_PATH, name="G2_chapter6"))
        self.robot.initialize()
        self.cube.initialize()
        self.arm_base = SingleXFormPrim("/genie/arm_base_link")
        self.tcp = SingleXFormPrim("/genie/gripper_r_center_link")
        self.chassis = SingleXFormPrim("/genie/chassis_link")
        if not self.config.headless:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([3.0, -3.1, 2.3]), target=np.array([.9, -.2, .75]))
        for _ in range(120):
            self.step()
        # 使用底盘刚体的位姿，而非可能位于躯干上的 articulation 根。
        print(f"[环境] G2 {self.robot.num_dof} DOF，底盘初始位姿 {self.base_pose()}", flush=True)

    def base_pose(self):
        p, q = self.chassis.get_world_pose()
        w, x, y, z = q
        return Pose2D(float(p[0]), float(p[1]), math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))

    def world_to_arm(self, position, rotation):
        p, q = self.arm_base.get_world_pose()
        r = quaternion_to_matrix(q)
        return r.T @ (np.asarray(position)-p), r.T @ rotation

    def tcp_pose(self):
        p, q = self.tcp.get_world_pose()
        return np.asarray(p, dtype=float), quaternion_to_matrix(q)

    def step(self):
        if not self.app.is_running() or not self.world.is_playing():
            raise RuntimeError("仿真已停止；请重新运行示例，不在任务中途暂停/重置")
        self.world.step(render=not self.config.headless)
        self.steps += 1

    def close(self):
        if self.app is not None:
            self.app.close()
            self.app = None
