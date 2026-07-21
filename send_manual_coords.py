#!/usr/bin/env python3
"""
send_manual_coords.py

Sends a manually-specified target position to the JAKA Mini 2 arm running
in simulation (RViz-only sim mode or Gazebo), via the FollowJointTrajectory
action exposed by the jaka_planner package.

This is a first manual test step: it does NOT do vision-based alignment yet.
It simply proves that a Python script can command the arm to a target
position in the simulated environment (per MP-08, Week 2 objective).

Prerequisites (run these in separate terminals BEFORE running this script):
    Terminal 1 (RViz-only simulation, no physical robot):
        ros2 launch jaka_<robot_model>_moveit_config demo.launch.py use_rviz_sim:=true

    OR Terminal 1 (Gazebo simulation):
        ros2 launch jaka_<robot_model>_moveit_config demo_gazebo.launch.py

Replace <robot_model> with your actual JAKA model config package
(check with: ros2 pkg list | grep jaka)
"""

import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# ---------------------------------------------------------------------------
# CONFIGURATION — adjust these for your setup
# ---------------------------------------------------------------------------
ROBOT_MODEL = "minicobo"  # e.g. "zu3", "s5", "a12", "minicobo" -- match your moveit_config package

# The 6 joint names for a JAKA 6-DOF arm. Verify with:
#   ros2 topic echo /joint_states
JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

# Manually defined target joint positions (radians).
# This is the "manual coordinates" step -- later this will be replaced by
# the output of the pixel -> real-world (TF2) conversion from the vision pipeline.
TARGET_JOINT_POSITIONS = [0.0, 0.5, -0.5, 0.0, 1.0, 0.0]

# Time (seconds) allowed to reach the target -- tune for smoothness
TIME_TO_REACH_SECONDS = 4


class ManualCoordSender(Node):
    def __init__(self):
        super().__init__('manual_coord_sender')
        action_name = f'/jaka_{ROBOT_MODEL}_controller/follow_joint_trajectory'
        self.get_logger().info(f'Connecting to action server: {action_name}')
        self._client = ActionClient(self, FollowJointTrajectory, action_name)

    def send_target(self, joint_positions, duration_sec):
        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Action server not available. Is the simulation running?')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = duration_sec

        goal.trajectory.points = [point]

        self.get_logger().info(f'Sending target joint positions: {joint_positions}')
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the action server.')
            return False

        self.get_logger().info('Goal accepted, waiting for execution to complete...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('Movement complete.')
        return True


def main():
    rclpy.init()
    node = ManualCoordSender()
    try:
        node.send_target(TARGET_JOINT_POSITIONS, TIME_TO_REACH_SECONDS)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
