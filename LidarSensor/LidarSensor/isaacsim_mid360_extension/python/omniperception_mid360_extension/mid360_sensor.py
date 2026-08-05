import numpy as np
import torch
import warp as wp
import omni.usd
from pxr import UsdGeom
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType


class Mid360Sensor:
    def __init__(self, stage, prim_path: str, config: dict):
        self.stage = stage
        self.prim_path = prim_path
        self.prim = stage.GetPrimAtPath(prim_path)
        if not self.prim or not self.prim.IsA(UsdGeom.Xform):
            raise ValueError(f"Prim {prim_path} does not exist or is not an Xform")

        self.link_path = prim_path + "/LidarLink"
        self.link_prim = None

        self.config = config or {}
        self.device = self.config.get("device", "cuda:0")
        self.num_envs = 1
        self.num_sensors = 1

        self.lidar_config = LidarConfig(
            sensor_type=LidarType.MID360,
            max_range=self.config.get("max_range", 30.0),
            min_range=self.config.get("min_range", 0.2),
            return_pointcloud=self.config.get("return_pointcloud", True),
            pointcloud_in_world_frame=self.config.get("pointcloud_in_world_frame", True),
            enable_sensor_noise=self.config.get("enable_sensor_noise", False),
            update_frequency=self.config.get("update_frequency", 25.0),
        )

        self._build_sensor_env()
        self._create_lidar_link()
        self._create_mid360_reference()
        self._create_visualization()
        self._create_lidar_sensor()

    def _build_sensor_env(self):
        self.sensor_env = {
            "num_envs": self.num_envs,
            "mesh_ids": self.config.get("mesh_ids", 0),
            "sensor_pos_tensor": torch.zeros((self.num_envs, 3), device=self.device),
            "sensor_quat_tensor": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=self.device),
        }

    def _create_lidar_link(self):
        self.link_prim = self.stage.GetPrimAtPath(self.link_path)
        if self.link_prim and self.link_prim.IsA(UsdGeom.Xform):
            return

        self.link_prim = self.stage.DefinePrim(self.link_path, "Xform")
        if not self.link_prim:
            raise RuntimeError(f"Failed to create LidarLink prim at {self.link_path}")

    def _create_mid360_reference(self):
        mid360_usd_path = self.config.get("mid360_usd_path")
        if not mid360_usd_path:
            return

        child_path = self.link_path + "/Mid360"
        child_prim = self.stage.GetPrimAtPath(child_path)
        if child_prim and child_prim.HasAuthoredReferences():
            return

        child_prim = self.stage.DefinePrim(child_path, "Xform")
        if child_prim:
            child_prim.GetReferences().AddReference(mid360_usd_path)

    def _create_visualization(self):
        sphere = UsdGeom.Sphere.Define(self.stage, self.link_path + "/visual")
        sphere.CreateRadiusAttr(0.05)
        sphere.CreateDisplayColorAttr([(0.0, 0.6, 1.0)])
        sphere.CreateTranslateAttr((0.0, 0.0, 0.0))

    def _create_lidar_sensor(self):
        self.lidar = LidarSensor(
            env=self.sensor_env,
            env_cfg={},
            sensor_config=self.lidar_config,
            num_sensors=self.num_sensors,
            device=self.device,
        )

    def _read_prim_pose(self):
        prim = self.link_prim if self.link_prim else self.prim
        xform = UsdGeom.XformCommonAPI(prim)
        trans = xform.GetTranslateAttr().Get() or (0.0, 0.0, 0.0)
        rot = xform.GetRotateAttr().Get() or (0.0, 0.0, 0.0)
        trans = np.array(trans, dtype=np.float32)
        rot = np.array(rot, dtype=np.float32)
        return trans, self._euler_to_quat(rot)

    def _euler_to_quat(self, euler_deg):
        r = np.deg2rad(euler_deg)
        cy = np.cos(r[2] * 0.5)
        sy = np.sin(r[2] * 0.5)
        cp = np.cos(r[1] * 0.5)
        sp = np.sin(r[1] * 0.5)
        cr = np.cos(r[0] * 0.5)
        sr = np.sin(r[0] * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float32)

    def update_pose(self):
        trans, quat = self._read_prim_pose()
        self.sensor_env["sensor_pos_tensor"][0] = torch.from_numpy(trans)
        self.sensor_env["sensor_quat_tensor"][0] = torch.from_numpy(quat)
        self.lidar.lidar_positions = wp.from_torch(
            self.sensor_env["sensor_pos_tensor"].view(self.num_envs, 1, 3), dtype=wp.vec3
        )
        self.lidar.lidar_quat_array = wp.from_torch(
            self.sensor_env["sensor_quat_tensor"].view(self.num_envs, 1, 4), dtype=wp.quat
        )

    def step(self):
        self.update_pose()
        return self.lidar.update()

    def shutdown(self):
        pass
