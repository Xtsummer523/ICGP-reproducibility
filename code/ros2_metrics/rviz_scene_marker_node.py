from __future__ import annotations

from collections import deque
import math

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from icgp_experiments.config import ExperimentSpec, RobotSpec, load_experiment_spec
from icgp_experiments.qos import sensor_qos
from icgp_experiments.ros_utils import odom_xy_yaw, pose_stamped_xy_yaw


PALETTE = [
    (0.050, 0.280, 0.780, 1.0),
    (0.900, 0.220, 0.120, 1.0),
    (0.100, 0.620, 0.260, 1.0),
    (0.600, 0.250, 0.780, 1.0),
    (0.920, 0.560, 0.050, 1.0),
    (0.000, 0.600, 0.650, 1.0),
    (0.780, 0.120, 0.430, 1.0),
    (0.300, 0.300, 0.300, 1.0),
]


class RvizSceneMarkerNode(Node):
    """Publish live RViz markers for visual support screenshots.

    This node is intentionally outside the metric path. It subscribes to the
    same pose streams used by operators and renders robots, goals, safety
    margins, walls, and short trails for RViz. Paper metrics remain computed
    from CSV/JSON/rosbag logs, not from marker geometry or screenshots.
    """

    def __init__(self) -> None:
        super().__init__("icgp_rviz_scene_marker_node")
        self.declare_parameter("config_path", "")
        self.declare_parameter("trail_length", 120)
        self.declare_parameter("publish_rate_hz", 5.0)

        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        if not config_path:
            raise ValueError("config_path parameter is required")
        self.spec = load_experiment_spec(config_path)
        self.trail_length = max(2, int(self.get_parameter("trail_length").value))
        publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))

        self.last_pose: dict[str, tuple[float, float, float]] = {}
        self.trails: dict[str, deque[tuple[float, float]]] = {
            robot.robot_id: deque(maxlen=self.trail_length) for robot in self.spec.robots
        }
        self.subscriptions_ = []
        for robot in self.spec.robots:
            if robot.pose_topic:
                self.subscriptions_.append(
                    self.create_subscription(PoseStamped, robot.pose_topic, lambda msg, r=robot: self._on_pose(msg, r), sensor_qos())
                )
            else:
                self.subscriptions_.append(
                    self.create_subscription(Odometry, robot.odom_topic, lambda msg, r=robot: self._on_odom(msg, r), sensor_qos())
                )

        self.marker_pub = self.create_publisher(MarkerArray, "/icgp/rviz_scene_markers", 1)
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_world_to_map_tf()
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish_markers)
        self.get_logger().info(
            f"RViz marker node rendering {len(self.spec.robots)} robots for scenario={self.spec.scenario}; "
            "visual support only, not metric source"
        )

    def _publish_world_to_map_tf(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = self.spec.map_frame
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.w = 1.0
        self.static_tf.sendTransform(transform)

    def _on_pose(self, msg: PoseStamped, robot: RobotSpec) -> None:
        x, y, yaw = pose_stamped_xy_yaw(msg)
        self._store_pose(robot.robot_id, x, y, yaw)

    def _on_odom(self, msg: Odometry, robot: RobotSpec) -> None:
        x, y, yaw = odom_xy_yaw(msg)
        self._store_pose(robot.robot_id, x, y, yaw)

    def _store_pose(self, robot_id: str, x: float, y: float, yaw: float) -> None:
        self.last_pose[robot_id] = (x, y, yaw)
        trail = self.trails[robot_id]
        if not trail or math.hypot(x - trail[-1][0], y - trail[-1][1]) > 0.01:
            trail.append((x, y))

    def _base_marker(self, marker_id: int, ns: str, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.spec.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _set_color(self, marker: Marker, color: tuple[float, float, float, float]) -> None:
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

    def _wall_markers(self) -> list[Marker]:
        boxes: list[tuple[str, float, float, float, float]] = []
        if self.spec.scenario == "merge_bottleneck":
            boxes = [("gate_top", 0.0, 1.0, 1.4, 0.25), ("gate_bottom", 0.0, -1.0, 1.4, 0.25)]
        else:
            boxes = [("left_wall", 0.0, 1.05, 4.0, 0.25), ("right_wall", 0.0, -1.05, 4.0, 0.25)]
        markers: list[Marker] = []
        for idx, (_name, x, y, sx, sy) in enumerate(boxes):
            marker = self._base_marker(100 + idx, "walls", Marker.CUBE)
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.25
            marker.scale.x = sx
            marker.scale.y = sy
            marker.scale.z = 0.50
            self._set_color(marker, (0.24, 0.24, 0.24, 0.92))
            markers.append(marker)
        return markers

    def _goal_marker(self, robot: RobotSpec, idx: int, color: tuple[float, float, float, float]) -> Marker:
        marker = self._base_marker(200 + idx, "goals", Marker.CYLINDER)
        marker.pose.position.x = robot.goal_xy[0]
        marker.pose.position.y = robot.goal_xy[1]
        marker.pose.position.z = 0.015
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.03
        self._set_color(marker, (color[0], color[1], color[2], 0.45))
        return marker

    def _text_marker(self, robot: RobotSpec, idx: int, x: float, y: float, color: tuple[float, float, float, float]) -> Marker:
        marker = self._base_marker(300 + idx, "robot_labels", Marker.TEXT_VIEW_FACING)
        label_x, label_y = self._label_position(x, y)
        marker.pose.position.x = label_x
        marker.pose.position.y = label_y
        marker.pose.position.z = 0.62
        marker.scale.z = 0.12
        marker.text = f"R{idx + 1}"
        self._set_color(marker, (0.02, 0.02, 0.02, 1.0))
        return marker

    def _label_position(self, x: float, y: float) -> tuple[float, float]:
        candidates = [
            (0.42, 0.0),
            (-0.42, 0.0),
            (0.0, 0.42),
            (0.0, -0.42),
            (0.32, 0.32),
            (-0.32, 0.32),
            (0.32, -0.32),
            (-0.32, -0.32),
        ]
        goals = [robot.goal_xy for robot in self.spec.robots]
        best = (x + 0.42, y, -1.0)
        for dx, dy in candidates:
            lx = x + dx
            ly = y + dy
            bound_margin = min(lx + 2.45, 2.45 - lx, ly + 1.65, 1.65 - ly)
            if bound_margin < 0.0:
                continue
            goal_margin = min((math.hypot(lx - gx, ly - gy) for gx, gy in goals), default=1.0)
            wall_penalty = 0.0
            if self.spec.scenario == "merge_bottleneck":
                for wx, wy in [(0.0, 1.0), (0.0, -1.0)]:
                    if abs(lx - wx) < 0.85 and abs(ly - wy) < 0.28:
                        wall_penalty += 1.0
            score = 1.4 * bound_margin + 0.8 * goal_margin - wall_penalty
            if score > best[2]:
                best = (lx, ly, score)
        return best[0], best[1]

    def _robot_markers(self, robot: RobotSpec, idx: int, x: float, y: float, yaw: float) -> list[Marker]:
        color = PALETTE[idx % len(PALETTE)]
        markers: list[Marker] = []

        safety = self._base_marker(400 + idx, "safety_disks", Marker.CYLINDER)
        safety.pose.position.x = x
        safety.pose.position.y = y
        safety.pose.position.z = 0.006
        safety.scale.x = self.spec.safety_distance_m
        safety.scale.y = self.spec.safety_distance_m
        safety.scale.z = 0.012
        self._set_color(safety, (color[0], color[1], color[2], 0.14))
        markers.append(safety)

        body = self._base_marker(500 + idx, "robots", Marker.CYLINDER)
        body.pose.position.x = x
        body.pose.position.y = y
        body.pose.position.z = 0.08
        body.scale.x = 2.0 * self.spec.robot_radius_m
        body.scale.y = 2.0 * self.spec.robot_radius_m
        body.scale.z = 0.12
        self._set_color(body, color)
        markers.append(body)

        arrow = self._base_marker(600 + idx, "headings", Marker.ARROW)
        arrow.points = [
            Point(x=x, y=y, z=0.18),
            Point(x=x + 0.34 * math.cos(yaw), y=y + 0.34 * math.sin(yaw), z=0.18),
        ]
        arrow.scale.x = 0.035
        arrow.scale.y = 0.075
        arrow.scale.z = 0.075
        self._set_color(arrow, (0.02, 0.02, 0.02, 1.0))
        markers.append(arrow)

        trail_points = list(self.trails[robot.robot_id])
        if len(trail_points) >= 2:
            trail = self._base_marker(700 + idx, "trails", Marker.LINE_STRIP)
            trail.scale.x = 0.035
            trail.points = [Point(x=px, y=py, z=0.035) for px, py in trail_points]
            self._set_color(trail, (color[0], color[1], color[2], 0.90))
            markers.append(trail)

        markers.append(self._text_marker(robot, idx, x, y, color))
        return markers

    def _publish_markers(self) -> None:
        markers = self._wall_markers()
        for idx, robot in enumerate(self.spec.robots):
            markers.append(self._goal_marker(robot, idx, PALETTE[idx % len(PALETTE)]))
            pose = self.last_pose.get(robot.robot_id)
            if pose is None:
                continue
            markers.extend(self._robot_markers(robot, idx, *pose))
        self.marker_pub.publish(MarkerArray(markers=markers))


def main() -> None:
    rclpy.init()
    node = RvizSceneMarkerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
