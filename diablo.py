from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdPhysics
from pxr import PhysxSchema
from typing import Optional, Union

class Diablo(Robot):

    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "diablo",
        usd_path: Optional[str] = "./diablo.usd",
        translation: Optional[Union[list, np.array, torch.tensor]] = torch.tensor([1.0, 0.0, 0.0]),
        orientation: Optional[Union[list, np.array, torch.tensor]] = torch.tensor([0.0, 0.0, 0.0, 1.0]),
    ) -> None:
        '''Initializes the Diablo robot.

        Args:
            - prim_path: path to the robot in the stage
            - name: name of the robot
            - usd_path: path to the USD file of the robot
            - translation: initial translation of the robot
            - orientation: initial orientation of the robot
        '''
        self._usd_path = usd_path
        self._name = name
        self._position = torch.tensor(translation)
        self._orientation = torch.tensor(orientation)

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None #
        )

        self.indices = { # very well ordered and named joint indices
            "left_base": 1, # hip
            "right_base": 0,
            "left_leg": 3, # placeholder for knee motor on real robot; here they do the same as the _base motors
            "right_leg": 6,
            "left_knee": 2, # knee; does not exist on the real robot, where knee is moved from base via parallelogram joints
            "right_knee": 5,
            "left_wheel": 4, # wheels
            "right_wheel": 7
        }

        dof_paths = [
            "/base_link/Rev1",
            "/base_link/Rev2",
            "/motor_left_link_1/Rev3",
            "/leg_left_link_1/Rev4", 
            "/leg2_left_link_1/Rev5",
            "/motor_right_link_1/Rev6",
            "/leg_right_link_1/Rev7",
            "/leg2_right_link_1/Rev8"
        ]

        # to change, values are in SI units
        drive_type = ["angular"] * 8
        self.target_type = ["velocity"] * 8
        default_dof_pos = [0.] * 8
        stiffness = [15000] * 8
        damping = [1] * 8
        max_force = [1000] * 8
        max_velocity = [1000] * 8

        self.joints = []
        for i, dof in enumerate(dof_paths):
            joint_prim = get_prim_at_path(f"{self.prim_path}{dof}")
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type[i])

            if self.target_type == "position":
                if not drive.GetTargetPositionAttr(): drive.CreateTargetPositionAttr(default_dof_pos[i])
                else: drive.GetTargetPositionAttr().Set(default_dof_pos[i])
            elif self.target_type == "velocity":
                if not drive.GetTargetVelocityAttr(): drive.CreateTargetVelocityAttr(default_dof_pos[i])
                else: drive.GetTargetVelocityAttr().Set(default_dof_pos[i])
            if not drive.GetStiffnessAttr(): drive.CreateStiffnessAttr(stiffness[i])
            else: drive.GetStiffnessAttr().Set(stiffness[i])
            if not drive.GetDampingAttr(): drive.CreateDampingAttr(damping[i])
            else: drive.GetDampingAttr().Set(damping[i])
            if not drive.GetMaxForceAttr(): drive.CreateMaxForceAttr(max_force[i])
            else: drive.GetMaxForceAttr().Set(max_force[i])
            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])

            self.joints.append(drive)
    
    def set_joint_velocity(self, index: Union[int, str], target: float):
        '''Sets the velocity target for a joint in the robot.

        Indices are either ints or the following strings (in order of indices):
            right_base, left_base, left_knee, left_leg, left_wheel, right_knee, right_leg, right_wheel
        Args:
            - index: index of the joint to set the velocity target
            - target: target velocity for the joint
        '''
        if isinstance(index, str):
            index = self.indices[index]
        if self.target_type[index] == "position":
            self.joints[index].GetTargetPositionAttr().Set(target)
        elif self.target_type[index] == "velocity":
            self.joints[index].GetTargetVelocityAttr().Set(target)

    def set_joint_velocities(self, target: Optional[Union[list, np.array]]) -> None:
        '''Sets the velocity target for all joints in the robot.

        Joint order is: right_base, left_base, left_knee, left_leg, left_wheel, right_knee, right_leg, right_wheel
        Args:
            - target: list of target velocities for each joint in the robot
        '''
        for i in len(self.joints):
            self.set_joint_velocity(i, target[i])

    def set_joint_position(self, index: Union[int, str], position):
        '''Sets the position target for a joint in the robot.

        Indices are either ints or the following strings (in order of indices):
            right_base, left_base, left_knee, left_leg, left_wheel, right_knee, right_leg, right_wheel
        Args:
            - index: index of the joint to set the position target
            - position: target position for the joint
        '''
        if isinstance(index, str):
            index = self.indices[index]
        self.joints[index].GetTargetPositionAttr().Set(position)

    def set_joint_positions(self, position: Optional[Union[list, np.array]]) -> None:
        '''Sets the position target for all joints in the robot.

        Joint order is: right_base, left_base, left_knee, left_leg, left_wheel, right_knee, right_leg, right_wheel
        Args:
            - position: list of target positions for each joint in the robot
        '''
        for i in len(self.joints):
            self.set_joint_position(i, position[i])