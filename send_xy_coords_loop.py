#!/usr/bin/env python3
"""
send_xy_coords_loop.py

Simulates the future camera -> robot pipeline:
    - The camera will eventually send (x, y) pixel-based coordinates.
    - Only the first two joints (joint_1, joint_2) respond to (x, y).
    - The remaining joints (3-6) stay at 0.

For now, (x, y) are RANDOM values, sent every 4 seconds, just to test
that the simulated arm correctly moves only its first two joints.

Later, x and y will come from app.py (YOLO detection) instead of random().

Prerequisite (run in another terminal first):
    ros2 launch jaka_minicobo_moveit_config demo.launch.py use_rviz_sim:=true

Stop this script anytime with Ctrl+C.
"""

import random
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

ROBOT_MODEL = "minicobo"

JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

# Safe random range in radians for joint_1 and joint_2 (roughly -90deg to +90deg).
# Adjust if the arm hits its limits or looks too extreme.
XY_MIN = -1.5
XY_MAX = 1.5

TIME_TO_REACH_SECONDS = 3   # how long each move takes
PAUSE_BETWEEN_MOVES = 4     # seconds between each new (x, y) send


class XYCoordSender(Node):
    def __init__(self):
        super().__init__('xy_coord_sender')
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

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('Movement complete.')
        return True


def build_joint_positions(x, y):
    """Only joint_1 (from x) and joint_2 (from y) move. Joints 3-6 stay at 0."""
    return [x, y, 0.0, 0.0, 0.0, 0.0]


def main():
    rclpy.init()
    node = XYCoordSender()
    try:
        while rclpy.ok():
            # --- placeholder for real camera (x, y) later ---
            x = round(random.uniform(XY_MIN, XY_MAX), 2)
            y = round(random.uniform(XY_MIN, XY_MAX), 2)
            # --------------------------------------------------

            target = build_joint_positions(x, y)
            node.get_logger().info(f'Simulated camera coords -> x={x}, y={y}')
            node.send_target(target, TIME_TO_REACH_SECONDS)
            time.sleep(PAUSE_BETWEEN_MOVES)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
