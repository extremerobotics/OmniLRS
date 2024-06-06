import hydra
import numpy as np
import sys, os, time
from omegaconf import DictConfig
from run import omegaconfToDict, instantiateConfigs
from scipy.spatial.transform import Rotation

headless = False
diablo_position = (3, 2.5, 0.01) # good spot for lunalab and lunaryard
diablo_rotation = (0, 0, -75)
camera_position = (2.4, 1, 0.7)

quantum_efficiency = 1e3
dark_count_rate = 0.01
dt = 1/500. # determines simulation dt and camera framerate
num_average = 1

### SIM SETUP
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

if __name__ == "__main__":
    run() # just for the hydra wrapper

    ### DIABLO
    rot = Rotation.from_euler("xyz", diablo_rotation, degrees=True).as_quat()
    rot = [rot[3], rot[0], rot[1], rot[2]]
    from diablo import Diablo
    diablo_stage_path = "/diablo"
    diablo = Diablo(
        prim_path=diablo_stage_path,
        name="diablo",
        usd_path="./diablo.usda",
        translation=diablo_position,
        orientation=rot # (w, x, y, z) quaternion
    )

    from omni.isaac.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array(camera_position), target=np.array(diablo_position)) # sets viewport

    ### DIABLO CAMERA
    from omni.isaac.sensor import Camera
    diablo_camera_path = diablo_stage_path + "/base_link/camera" # attached to the main body
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

    from spad import SPADWriter
    rep.WriterRegistry.register(SPADWriter)
    spad_writer: SPADWriter = rep.WriterRegistry.get("SPADWriter")
    spad_writer.initialize(
        output_path=os.path.join(os.getcwd(), "images"),
        dt=dt,
        quantum_efficiency=quantum_efficiency,
        dark_count_rate=dark_count_rate,
        num_average=num_average
    )
    spad_writer.attach(rp)

    # rgb_writer = rep.WriterRegistry.get("BasicWriter")
    # rgb_writer.initialize(
    #     output_dir=os.path.join(os.getcwd(), "images"),
    #     rgb=True
    # )
    # rgb_writer.attach(rp)

    # anns = []
    # anns.append(rep.AnnotatorRegistry.get_annotator("PtSelfIllumination"))
    # anns.append(rep.AnnotatorRegistry.get_annotator("PtDirectIllumation"))
    # anns.append(rep.AnnotatorRegistry.get_annotator("PtGlobalIllumination"))
    # for ann in anns: ann.attach(rp)

    ### DIABLO IMU
    from omni.isaac.sensor import IMUSensor
    diablo_imu = IMUSensor(
        prim_path=diablo_stage_path + "/base_link/imu_sensor", # attached to the main body
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
    world.reset()
    diablo.initialize()
    diablo_camera.initialize()
    diablo_imu.initialize()
    timeline.play()
    world.step(render=False)

    # i = 0
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

            ### sensor data
            # print(diablo_imu.get_current_frame()) # lin_acc, ang_vel, orientation
            # frame = spad_writer.get_frame() # last SPAD averaged frame
            # subframe = spad_writer.get_subframe() # last SPAD frame

            ### robot control
            # diablo.set_joint_velocity("left_wheel", i % 200 - 100)
            # diablo.set_joint_velocity("right_wheel", 100 - i % 200)
            # i += 1

            ### debug annotators
            # from PIL import Image
            # for j in range(3): # self, direct, global
            #     data = anns[j].get_data() * 255
            #     data = data[:, :, :-1].astype(np.uint8)
            #     Image.fromarray(data, 'RGB').save(f"images/debug{j}_{stime}.png")
            # data = anns[0].get_data() + anns[1].get_data() + anns[2].get_data()
            # data = np.mean(data[:, :, :-1], axis=2, dtype=np.double)
            # data = (data * 255).astype(np.uint8)
            # Image.fromarray(data, 'L').save(f"images/debug_{stime}.png")

            dtime2 = time.time()
            print(f"Sim-time: {str(stime):<10}Frame-time: {dtime2 - dtime:.5f}")
            dtime = dtime2
            stime = round(stime + dt, 4)

    timeline.stop()
    simulation_app.close()