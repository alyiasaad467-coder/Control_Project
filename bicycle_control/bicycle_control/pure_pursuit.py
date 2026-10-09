"""
High-Level Lateral Steering Controller: Geometric Pure Pursuit.
Calculates steering curvature from lookahead arc geometry.
"""

import math  # noqa: F401
import numpy as np  # noqa: F401


class PurePursuitController:
    """Adaptive Pure Pursuit lateral controller."""

    def __init__(self, wheelbase=1.25, kv=0.25, l_min=0.8, l_max=2.5,
                 max_steer_rad=math.radians(35.0)):
        self.L = wheelbase
        self.kv = kv
        self.l_min = l_min
        self.l_max = l_max
        self.max_steer_rad = max_steer_rad

    def compute_lookahead(self, v):
        """Adaptive lookahead distance: Ld = clip(kv * v + l_min, l_min, l_max)."""
        # TODO: Milestone 5.3 Step 1 — Adaptive Lookahead Horizon
        # The car looks further ahead at higher speeds to plan smoother turns.
        # Implement the speed-scaled lookahead formula and clamp it to the allowed range.

        lookahead = self.kv * v + self.l_min

        return float(np.clip(
            lookahead,
            self.l_min,
            self.l_max
        ))

    def find_target_waypoint(self, x, y, path_points, lookahead):
        """Searches along path for the target waypoint at lookahead distance."""
        # TODO: Milestone 5.3 Step 2 — Target Waypoint Selection
        # This selects the goal point the car will steer toward.
        # Find the nearest waypoint on the path, then walk forward until
        # you reach one that is at least 'lookahead' meters away.

        path_points = np.asarray(path_points)

        distances = np.sqrt(
            (path_points[:, 0] - x) ** 2
            + (path_points[:, 1] - y) ** 2
        )

        nearest_idx = int(np.argmin(distances))

        for i in range(nearest_idx, len(path_points)):
            distance = math.sqrt(
                (path_points[i, 0] - x) ** 2
                + (path_points[i, 1] - y) ** 2
            )

            if distance >= lookahead:
                return path_points[i]

        return path_points[-1]

    def compute_steering(self, x, y, yaw, target_pt, lookahead):
        """Computes steering angle in radians using Pure Pursuit geometry."""
        # TODO: Milestone 5.3 Steps 3 & 4 — Coordinate Transformation & Arc Law
        # This is the core of Pure Pursuit: transform the target into the vehicle's
        # local frame, then use the arc geometry formula to compute the steering angle.

        dx = target_pt[0] - x
        dy = target_pt[1] - y

        x_local = math.cos(yaw) * dx + math.sin(yaw) * dy
        y_local = -math.sin(yaw) * dx + math.cos(yaw) * dy

        curvature = (2.0 * y_local) / (lookahead ** 2)

        steering = math.atan(
            self.L * curvature
        )

        steering = float(np.clip(
            steering,
            -self.max_steer_rad,
            self.max_steer_rad
        ))

        return steering
