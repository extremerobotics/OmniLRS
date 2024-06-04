import hydra, sys, time
import numpy as np
from omegaconf import DictConfig
from run import omegaconfToDict, instantiateConfigs

headless = False
dt = 1/500.
diablo_position = (3, 2.5, 0.1) # good spot for lunalab and lunaryard
camera_position = (2.4, 1, 0.7)

@hydra.main(config_name="config", config_path="cfg")
def run(cfg: DictConfig):
    # hydra config for OmniLRS
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
    if use_omnilrs: # moon environment
        from src.environments_wrappers import startSim
        SM, simulation_app = startSim(cfg, simulation_app=simulation_app, dt=dt) # also handles ROS2
        world = SM.world
        timeline = SM.timeline
    else: # default environment
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

    ### IMPORT DIABLO
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

    world.reset()
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

            i += 1

            dtime2 = time.time()
            print(f"Sim-time: {str(stime):<10}Frame-time: {dtime2 - dtime:.5f}")
            dtime = dtime2
            stime = round(stime + dt, 4)

    timeline.stop()
    simulation_app.close()

if __name__ == "__main__":
    run()