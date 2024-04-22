import hydra
import numpy as np
from omegaconf import DictConfig
from run import omegaconfToDict, instantiateConfigs
import sys
import os
import PIL

### SIM SETUP
@hydra.main(config_name="config", config_path="cfg")
def run(cfg: DictConfig):
    global simulation_app, world, timeline, SM, use_omnilrs, use_ros2
    cfg = omegaconfToDict(cfg)
    cfg = instantiateConfigs(cfg)
    from omni.isaac.kit import SimulationApp # don't include anything else from omni before creating the SimulationApp
    use_omnilrs = np.any(["environment" in arg for arg in sys.argv])
    use_ros2 = cfg["mode"]["name"] == "ROS2"
    if use_omnilrs:
        from src.environments_wrappers import startSim
        SM, simulation_app = startSim(cfg) # also handles ROS2
        world = SM.world
        timeline = SM.timeline
    else:
        simulation_app = SimulationApp(cfg["rendering"]["renderer"].__dict__)
        if use_ros2:
            from src.environments_wrappers.ros2 import enable_ros2
            enable_ros2(simulation_app, bridge_name=cfg["mode"]["bridge_name"])
            import rclpy
            rclpy.init()
        SM = None
        import omni
        world = omni.isaac.core.World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        timeline = omni.timeline.get_timeline_interface()
    from omni.isaac.core import SimulationContext
    simulation_context = SimulationContext()
    simulation_context.set_simulation_dt(rendering_dt=1/60.)

run()

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

diablo_position = (3, 2.5, 0.1) # good spot for lunalab and lunaryard

def import_diablo():
    import omni.kit.commands
    import pathlib
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.distance_scale = 1
    file_path = pathlib.Path(__file__).parent.absolute()
    file_path = file_path.parent.absolute() / "diablo_ros2/diablo_visualise/diablo_simulation/urdf/diablo_simulation.urdf"
    status, diablo_stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=file_path,
        import_config=import_config,
    )
    omni.kit.commands.execute("IsaacSimTeleportPrim", prim_path=diablo_stage_path, translation=diablo_position, rotation=(0, 0, 0, 1))
    return diablo_stage_path

def get_diablo_joints():
    from pxr import UsdPhysics
    import omni.isaac.core.utils.stage as stage_utils
    stage = stage_utils.get_current_stage()
    joints = []
    for path in joint_paths:
        joints.append(UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/diablo_simulation" + path), "angular"))
        joints[-1].GetDampingAttr().Set(15000)
        joints[-1].GetStiffnessAttr().Set(0)
    return joints

diablo_stage_path = import_diablo()
joints = get_diablo_joints()
from omni.isaac.core.utils.viewports import set_camera_view
set_camera_view(eye=np.array([2.4, 1, 0.7]), target=np.array(diablo_position)) # sets viewport

### DIABLO CAMERA
from omni.isaac.sensor import Camera
diablo_camera_path = diablo_stage_path + "/base_link/camera"
diablo_camera = Camera(
    prim_path=diablo_camera_path,
    translation=(0.21, 0, 0.35), # front face of diablo
    resolution=(1280, 720),
    frequency=20,
)
diablo_camera.set_clipping_range(0.001, 100000)
diablo_camera.set_focal_length(.5)
out_dir = os.path.dirname(os.path.join(os.getcwd(), "images", ""))
os.makedirs(out_dir, exist_ok=True)

### CAMERA RENDERING
import omni.replicator.core as rep
rp = rep.create.render_product(camera=diablo_camera_path, resolution=(1280, 720), name="rp")
rep.settings.carb_settings(setting="rtx-transient.aov.enableRtxAovs", value=True)
rep.settings.carb_settings(setting="rtx-transient.aov.enableRtxAovsSecondary", value=True)
# import warp as wp
# @wp.kernel # runs on gpu
# def spad_kernel(data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array3d(dtype=wp.uint8)):
#     i, j = wp.tid()
#     state = wp.rand_init(42, wp.tid())
#     p = wp.randf(state)
#     photon_count = data_in[i, j, 0] + data_in[i, j, 1] + data_in[i, j, 2] / 768
#     quantum_efficiency = 0.5
#     dark_count_rate = 0.001
#     out = p > 2.71828**(-photon_count*quantum_efficiency - dark_count_rate)
#     data_out[i, j, 0] = out * 255
#     data_out[i, j, 1] = out * 255
#     data_out[i, j, 2] = out * 255
#     data_out[i, j, 3] = data_in[i, j, 3]

def spad_kernel(data_in: np.ndarray) -> np.ndarray: # runs on cpu
    out = np.zeros(data_in.shape, dtype=np.uint8)
    photon_count = np.sum(data_in[:, :, :3], axis=2) / 768
    quantum_efficiency = 0.5
    dark_count_rate = 0.001
    out[:, :, 0] = 255 * (np.random.rand(*photon_count.shape) > np.exp(-photon_count*quantum_efficiency - dark_count_rate)).astype(np.uint8)
    out[:, :, 1] = out[:, :, 0]
    out[:, :, 2] = out[:, :, 0]
    out[:, :, 3] = data_in[:, :, 3]
    return out

ann_names = ["rgb", "PtDirectIllumation", "PtGlobalIllumination"] # there is a typo in PtDirectIllumation
anns = []
for ann_name in ann_names:
    anns.append(rep.AnnotatorRegistry.get_annotator(ann_name))
    anns[-1].attach(rp)
import carb
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)
rep.AnnotatorRegistry.register_augmentation("spad_kernel", rep.annotators.Augmentation.from_function(spad_kernel))
anns.append(rep.AnnotatorRegistry.get_annotator("rgb"))
anns[-1].augment(rep.AnnotatorRegistry.get_augmentation("spad_kernel"))
anns[-1].attach(rp)
ann_names.append("SPAD")

### DIABLO IMU
from omni.isaac.sensor import IMUSensor
diablo_imu = IMUSensor(
    prim_path=diablo_stage_path + "/base_link/imu_sensor",
    translation=np.array([0, 0, 0]),
    frequency=60,
)

### ROS2 CAMERA STREAM
if use_ros2 and False:
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
diablo_camera.initialize()
diablo_imu.initialize()
timeline.play()
i = 0
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
        if i < 20:
            for j, ann in enumerate(anns):
                frame = ann.get_data()
                print(frame.shape)
                if frame.size != 0:
                    print(frame[-1])
                    # PIL.Image.fromarray(frame, "RGBA").save(f"{out_dir}/{ann_names[j]}_{i}.png")
                    PIL.Image.fromarray(frame[:, :, :3], "RGB").save(f"{out_dir}/{ann_names[j]}_{i}.png")
        i += 1

timeline.stop()
simulation_app.close()