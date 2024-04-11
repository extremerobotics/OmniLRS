import hydra
import numpy as np
from omegaconf import DictConfig
from run import omegaconfToDict, instantiateConfigs
import sys

### DIABLO CONSTANTS
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
    import omni.isaac.core.utils.stage as stage_utils
    from pxr import UsdPhysics
    stage = stage_utils.get_current_stage()
    joints = []
    for path in joint_paths:
        joints.append(UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/diablo_simulation" + path), "angular"))
        joints[-1].GetDampingAttr().Set(15000)
        joints[-1].GetStiffnessAttr().Set(0)
    return joints

@hydra.main(config_name="config", config_path="cfg")
def run(cfg: DictConfig):
    cfg = omegaconfToDict(cfg)
    cfg = instantiateConfigs(cfg)

    use_omnilrs = np.any(["environment" in arg for arg in sys.argv])
    if use_omnilrs:
        from src.environments_wrappers import startSim
        SM, simulation_app = startSim(cfg)
        world = SM.world
        timeline = SM.timeline
    else: # setup own simulation with ground plane
        from omni.isaac.kit import SimulationApp # don't include anything else from omni before creating the SimulationApp
        simulation_app = SimulationApp(cfg["rendering"]["renderer"].__dict__)
        if cfg["mode"]["name"] == "ROS2":
            from src.environments_wrappers.ros2 import enable_ros2
            enable_ros2(simulation_app, bridge_name=cfg["mode"]["bridge_name"])
            import rclpy
            rclpy.init()
        SM = None
        import omni
        world = omni.isaac.core.World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        timeline = omni.timeline.get_timeline_interface()
        
    diablo_stage_path = import_diablo()
    joints = get_diablo_joints()

    ### ADD DIABLO CAMERA
    from omni.isaac.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array([2.4, 1, 0.7]), target=np.array(diablo_position)) # sets viewport
    from omni.isaac.sensor import Camera
    diablo_camera = Camera(
        prim_path=diablo_stage_path + "/base_link/camera",
        translation=(0.21, 0, 0.35), # front face of diablo
        resolution=(720, 1280),
        frequency=20,
    )
    diablo_camera.set_clipping_range(0.001, 100000)
    diablo_camera.set_focal_length(5.)

    ### ADD DIABLO IMU
    from omni.isaac.sensor import IMUSensor
    imu_sensor = IMUSensor(
        prim_path=diablo_stage_path + "/base_link/imu_sensor",
        translation=np.array([0, 0, 0]),
        frequency=60,
    )

    ### RUN SIMULATION
    left_wheel_joint = joints[LEFT_WHEEL]
    right_wheel_joint = joints[RIGHT_WHEEL]
    i = 0
    world.reset()
    diablo_camera.initialize() # cameras need world.reset() and this
    timeline.play()
    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():
            if use_omnilrs and cfg["mode"]["name"] == "ROS2": # OmniLRS stuff
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
            print(imu_sensor.get_current_frame())
            left_wheel_joint.GetTargetVelocityAttr().Set(i - 100) # deg/time unit
            right_wheel_joint.GetTargetVelocityAttr().Set(100 - i)
            i = (i + 1) % 200
    
    timeline.stop()
    simulation_app.close()

if __name__ == "__main__":
    run()