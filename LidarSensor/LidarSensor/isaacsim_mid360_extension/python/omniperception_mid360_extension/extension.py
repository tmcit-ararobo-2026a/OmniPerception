import omni.ext
import omni.usd
from pxr import UsdGeom
from .mid360_sensor import Mid360Sensor

class OmniPerceptionMid360Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._stage = omni.usd.get_context().get_stage()
        self._mid360_sensors = {}
        print("[OmniPerception Mid360 Extension] started")

    def on_shutdown(self):
        for sensor in self._mid360_sensors.values():
            sensor.shutdown()
        self._mid360_sensors.clear()
        print("[OmniPerception Mid360 Extension] shutdown")

    def create_mid360_sensor(self, prim_path: str, config: dict = None):
        config = config or {}
        if prim_path in self._mid360_sensors:
            return self._mid360_sensors[prim_path]

        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Xform):
            raise ValueError(f"Prim {prim_path} does not exist or is not an Xform")

        sensor = Mid360Sensor(self._stage, prim_path, config)
        self._mid360_sensors[prim_path] = sensor
        return sensor

    def remove_mid360_sensor(self, prim_path: str):
        sensor = self._mid360_sensors.pop(prim_path, None)
        if sensor:
            sensor.shutdown()

    def get_mid360_sensor(self, prim_path: str):
        return self._mid360_sensors.get(prim_path)
