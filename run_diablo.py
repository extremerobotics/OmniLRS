import hydra
import numpy as np
from omegaconf import DictConfig
from run import omegaconfToDict, instantiateConfigs
import sys
import os
import time

### DIABLO AND SPAD STUFF
headless = False
dt = 1/500. # determines camera framerate
num_average = 20
diablo_position = (3, 2.5, 0.01) # good spot for lunalab and lunaryard
camera_position = (2.4, 1, 0.7)
quantum_efficiency = 0.5
dark_count_rate = 0.001

### SIM SETUP
@hydra.main(config_name="config", config_path="cfg")
def run(cfg: DictConfig):
    # config
    global simulation_app, world, timeline, SM, use_omnilrs, use_ros2
    cfg = omegaconfToDict(cfg)
    cfg = instantiateConfigs(cfg)
    use_omnilrs = np.any(["environment" in arg for arg in sys.argv])
    use_ros2 = cfg["mode"]["name"] == "ROS2"
    appcfg = cfg["rendering"]["renderer"].__dict__
    appcfg["headless"] = headless
    # simulation app
    from omni.isaac.kit import SimulationApp # don't include anything else from omni before creating the SimulationApp
    simulation_app = SimulationApp(appcfg)
    if use_omnilrs:
        from src.environments_wrappers import startSim
        SM, simulation_app = startSim(cfg, simulation_app=simulation_app, dt=dt) # also handles ROS2
        world = SM.world
        timeline = SM.timeline
    else:
        if use_ros2:
            from src.environments_wrappers.ros2 import enable_ros2
            enable_ros2(simulation_app, bridge_name=cfg["mode"]["bridge_name"])
            import rclpy
            rclpy.init()
        SM = None
        import omni
        world = omni.isaac.core.World(stage_units_in_meters=1.0, rendering_dt=dt, physics_dt=dt/2.)
        world.scene.add_default_ground_plane()
        timeline = omni.timeline.get_timeline_interface()

run() # just for the hydra wrapper

### DIABLO UTILS
# _BASE and _LEG do the same thing (the diablo robot only has 6 motors)
RIGHT_BASE = 0 # Rev1
LEFT_BASE = 1 # ...
LEFT_LEG = 2
LEFT_KNEE = 3
LEFT_WHEEL = 4
RIGHT_LEG = 5
RIGHT_KNEE = 6
RIGHT_WHEEL = 7 # Rev8

joint_paths = ["/base_link/Rev1", "/base_link/Rev2", "/motor_left_link_1/Rev3", "/leg_left_link_1/Rev4", 
               "/leg2_left_link_1/Rev5", "/motor_right_link_1/Rev6", "/leg_right_link_1/Rev7", "/leg2_right_link_1/Rev8"]

def get_diablo_joints(diablo_stage_path):
    from pxr import UsdPhysics
    import omni.isaac.core.utils.stage as stage_utils
    stage = stage_utils.get_current_stage()
    joints = []
    for path in joint_paths:
        joints.append(UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(diablo_stage_path + path), "angular"))
        joints[-1].GetDampingAttr().Set(15000)
        joints[-1].GetStiffnessAttr().Set(0)
    return joints

diablo_stage_path = "/diablo_simulation"
from omni.isaac.core.utils.stage import add_reference_to_stage
add_reference_to_stage("./diablo.usd", diablo_stage_path)
from omni.isaac.core.robots import Robot
diablo = Robot(prim_path=diablo_stage_path, name="diablo")
diablo.set_world_pose(
    position=np.array(diablo_position) + np.array([-0.05, 0, 0.05]), # second terms correct for diablo's weird origin
    orientation=np.array([1, 0, 0, 0]) + np.array([0.5, 0, -0.8660254, 0])) # (w, x, y, z) quaternion
from omni.isaac.core.utils.viewports import set_camera_view
set_camera_view(eye=np.array(camera_position), target=np.array(diablo_position)) # sets viewport
joints = get_diablo_joints(diablo_stage_path)

### DIABLO CAMERA
from omni.isaac.sensor import Camera
diablo_camera_path = diablo_stage_path + "/base_link/camera"
diablo_camera = Camera(
    prim_path=diablo_camera_path,
    translation=(0.21, 0, 0.67), # integration prototype (approx.)
    orientation=(0.966, 0, 0.259, 0), # quaternion, 30° down
    resolution=(1280, 720)
)
diablo_camera.set_clipping_range(0.001, 100)
diablo_camera.set_focal_length(.5)

### CAMERA RENDERING
import omni.replicator.core as rep
rp = rep.create.render_product(camera=diablo_camera_path, resolution=(1280, 720), name="rp")
rep.settings.carb_settings(setting="rtx-transient.aov.enableRtxAovs", value=True)
rep.settings.carb_settings(setting="rtx-transient.aov.enableRtxAovsSecondary", value=True)

from spad import SPADWriter
rep.WriterRegistry.register(SPADWriter)
spad_writer = rep.WriterRegistry.get("SPADWriter")
spad_writer.initialize(
    output_path=os.path.join(os.getcwd(), "images"),
    dt=dt,
    quantum_efficiency=quantum_efficiency,
    dark_count_rate=dark_count_rate,
    num_average=num_average
)
spad_writer.attach(rp)

### DIABLO IMU
from omni.isaac.sensor import IMUSensor
diablo_imu = IMUSensor(
    prim_path=diablo_stage_path + "/base_link/imu_sensor",
    translation=np.array([0, 0, 0]), # to be changed
)

### ROS2 CAMERA STREAM
if use_ros2 and False: # not working yet
    import omni.graph.core as og
    import usdrt.Sdf
    keys = og.Controller.Keys
    (ros_camera_graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": "/ROS_Camera",
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                ("getRenderProduct", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
                ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("createViewport.inputs:viewportId", 0),
                ("cameraHelperRgb.inputs:frameId", "sim_camera"),
                ("cameraHelperRgb.inputs:topicName", "rgb"),
                ("cameraHelperRgb.inputs:type", "rgb"),
                ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(diablo_camera_path)]),
            ],
        },
    )
    og.Controller.evaluate_sync(ros_camera_graph)
    simulation_app.update()

### RUN SIMULATION
left_wheel_joint = joints[LEFT_WHEEL]
right_wheel_joint = joints[RIGHT_WHEEL]
world.reset()
diablo.initialize()
diablo_camera.initialize()
diablo_imu.initialize()
timeline.play()
world.step(render=False)

i = 0
dtime = time.time()
stime = 0

while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        if use_omnilrs and use_ros2: # OmniLRS stuff
            if world.current_time_step_index == 0:
                world.reset()
                SM.ROSLabManager.reset()
                SM.ROSRobotManager.reset()
            SM.ROSLabManager.applyModifications()
            if SM.ROSLabManager.trigger_reset:
                SM.ROSRobotManager.reset()
                SM.ROSLabManager.trigger_reset = False
            SM.ROSRobotManager.applyModifications()

        ### CONTROL AND SENSING
        # print(diablo_imu.get_current_frame()) # lin_acc, ang_vel, orientation
        left_wheel_joint.GetTargetVelocityAttr().Set(i % 200 - 100) # deg/time unit
        right_wheel_joint.GetTargetVelocityAttr().Set(100 - i % 200)
        i += 1

        dtime2 = time.time()
        print(f"Sim-time: {stime}".ljust(20), f"dt: {dtime2 - dtime}")
        dtime = dtime2
        stime = round(stime + dt, 4)

timeline.stop()
simulation_app.close()