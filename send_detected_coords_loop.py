#!/usr/bin/env python3
"""
send_detected_coords_loop.py

Real pipeline (replaces send_xy_coords_loop.py's random values):
    - Subscribes to /detected_objects_xy (PoseArray) published by app.py.
    - Every 4 seconds, picks the NEXT detected object in round-robin order
      and sends its (x, y) to the simulated JAKA arm.
    - If there are 3 objects detected: obj1 -> wait 4s -> obj2 -> wait 4s ->
      obj3 -> wait 4s -> obj1 -> ... (a queue that cycles).
    - If nothing is currently detected, it just skips that tick and logs it.

Only joint_1 (from x) and joint_2 (from y) move. Joints 3-6 stay at 0,
same as send_xy_coords_loop.py.

Prerequisite (run in another terminal first):
    ros2 launch jaka_minicobo_moveit_config demo.launch.py use_rviz_sim:=true

Also run app.py (in its own terminal) so this node has something to
subscribe to.

Stop this script anytime with Ctrl+C.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseArray
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

ROBOT_MODEL = "minicobo"

JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

TIME_TO_REACH_SECONDS = 3   # how long each move takes
TICK_PERIOD_SECONDS = 4.0   # how often we send the next object in the queue

# --- pixel -> radian mapping ---------------------------------------------
# app.py's math coordinates are roughly in the range [-320, 320] for x and
# [-240, 240] for y on a 640x480 frame (origin at image center).
# Joints need small safe values, roughly [-1.5, 1.5] radians.
# This is a simple linear scale + clamp. Tune PIXEL_RANGE if your camera
# resolution is different, and JOINT_LIMIT if the arm looks too extreme.
PIXEL_RANGE = 320.0
JOINT_LIMIT = 1.5


def clamp(value, low, high):
    return max(low, min(high, value))


def pixel_to_joint(px):
    """Linearly scale a pixel coordinate to a safe joint angle in radians."""
    scaled = (px / PIXEL_RANGE) * JOINT_LIMIT
    return clamp(scaled, -JOINT_LIMIT, JOINT_LIMIT)


def build_joint_positions(x_pixel, y_pixel):
    """Only joint_1 (from x) and joint_2 (from y) move. Joints 3-6 stay at 0."""
    j1 = pixel_to_joint(x_pixel)
    j2 = pixel_to_joint(y_pixel)
    return [j1, j2, 0.0, 0.0, 0.0, 0.0]


class DetectedCoordSender(Node):
    def __init__(self):
        super().__init__('detected_coord_sender')

        action_name = f'/jaka_{ROBOT_MODEL}_controller/follow_joint_trajectory'
        self.get_logger().info(f'Connecting to action server: {action_name}')
        self._client = ActionClient(self, FollowJointTrajectory, action_name)

        # Latest list of (x, y) tuples seen from app.py. Always overwritten
        # by the most recent PoseArray -- it represents "what's visible now".
        self.latest_points = []

        # Round-robin index into latest_points.
        self.queue_index = 0

        self.subscription = self.create_subscription(
            PoseArray,
            '/detected_objects_xy',
            self.on_detections,
            10
        )

        # Fires send_next() every TICK_PERIOD_SECONDS, independent of how
        # fast frames arrive from app.py.
        self.timer = self.create_timer(TICK_PERIOD_SECONDS, self.send_next)

    def on_detections(self, msg):
        self.latest_points = [(pose.position.x, pose.position.y) for pose in msg.poses]

    def send_next(self):
        if not self.latest_points:
            self.get_logger().info('No objects currently detected, skipping this tick.')
            return

        # Wrap the index so it cycles: 0, 1, 2, 0, 1, 2, ...
        self.queue_index = self.queue_index % len(self.latest_points)
        x, y = self.latest_points[self.queue_index]
        self.get_logger().info(
            f'Queue position {self.queue_index + 1}/{len(self.latest_points)} '
            f'-> object at pixel ({x}, {y})'
        )
        self.queue_index += 1

        joint_positions = build_joint_positions(x, y)
        self.send_target(joint_positions, TIME_TO_REACH_SECONDS)

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
        future.add_done_callback(self.on_goal_response)
        return True

    def on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the action server.')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_result(self, future):
        self.get_logger().info('Movement complete.')


def main():
    rclpy.init()
    node = DetectedCoordSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
