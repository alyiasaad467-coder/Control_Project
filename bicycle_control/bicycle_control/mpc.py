"""
High-Level Lateral Steering Controller: Extended Kinematic Bicycle MPC.
Solves a constrained non-linear program over prediction horizon N using SciPy,
optimizing steering angle and longitudinal acceleration (mapped to throttle).
"""

import math  # noqa: F401
import numpy as np  # noqa: F401
from scipy.optimize import minimize  # noqa: F401


class KinematicBicycleMPC:
    """Nonlinear Model Predictive Control for an Extended Kinematic Bicycle Model.

    Optimizes future control sequences u = [delta_k, a_k] where steering angle delta_k
    and longitudinal acceleration a_k (mapped to throttle effort) are the control inputs,
    forward-simulating a 4-state extended kinematic bicycle model x = [x, y, theta, v]^T.
    """

    def __init__(self, wheelbase=1.25, dt=0.1, horizon=10,
                 max_steer_rad=math.radians(35.0), k_a=4.0,
                 max_accel=None, max_brake=None):
        self.L = wheelbase
        self.dt = dt
        self.N = horizon
        self.max_steer_rad = max_steer_rad
        self.k_a = float(max_accel if max_accel is not None else k_a)

        # Weights: heavily penalize lateral CTE, heading error, and steering rate
        self.w_lat = 30.0
        self.w_long = 1.0
        self.w_yaw = 10.0
        self.w_v = 1.0
        self.w_steer = 0.2
        self.w_dsteer = 6.0
        self.w_accel = 0.1

        self.last_u = np.zeros(2 * self.N)  # warm-start [delta_0, a_0, delta_1, a_1, ...]

    def solve(self, x0, ref_trajectory, current_steer=0.0):
        """Solves MPC optimization problem over horizon N.

        x0: [x, y, yaw, v]
        ref_trajectory: list of length N containing [x_ref, y_ref, yaw_ref, v_ref]
        current_steer: actual current steering angle in radians
        Returns: (steer_rad, throttle_cmd in [-1.0, 1.0])
        """
        # ======================================================================
        # TODO: Milestone 5.4 — Extended Kinematic Bicycle MPC
        #
        # 1. Horizon & Bounds Setup:
        #    - Determine effective horizon N = min(self.N, len(ref_trajectory)).
        #    - If N < 2, return (0.0, 0.0).
        #    - Construct variable bounds for the decision vector:
        #      u = [delta_0, a_0, delta_1, a_1, ..., delta_N-1, a_N-1]
        #      where delta_k in [-self.max_steer_rad, self.max_steer_rad] (steering input)
        #      and a_k in [-self.k_a, self.k_a] (longitudinal acceleration input).
        #
        # 2. Objective Function objective(u):
        #    - Unpack state [x, y, yaw, v] from x0 and set prev_delta = current_steer.
        #    - For each horizon step k in 0 .. N-1:
        #        a. Forward simulate state using discrete Extended Kinematic Bicycle equations
        #           (where longitudinal velocity v is an explicit state variable integrated
        #           forward with acceleration input a_k)
        #        b. Project tracking error into the path-aligned Frenet frame.
        #        c. Accumulate weighted quadratic costs:
        #           lateral CTE, heading error, speed error, steering, slew rate, accel.
        #        d. Update prev_delta = delta_k.
        #    - Return total cost.
        #
        # 3. Warm-Start Initialization:
        #    - Construct u_init by shifting self.last_u forward by 1 time step.
        #
        # 4. Numerical Optimization & Control Extraction:
        #    - Call scipy.optimize.minimize(objective, u_init, bounds=bounds,
        #                                   method='SLSQP',
        #                                   options={'maxiter': 25, 'ftol': 1e-3}).
        #    - Save optimal solution in self.last_u.
        #    - Extract first control step: delta_cmd = u*[0], accel_cmd = u*[1].
        #    - Map optimal acceleration a_0* to normalized throttle in [-1.0, 1.0]:
        #      throttle_cmd = accel_cmd / self.k_a
        #    - Return tuple: (delta_cmd, throttle_cmd).
        # ======================================================================

        N = min(self.N, len(ref_trajectory))

        if N < 2:
            return 0.0, 0.0

        ref = np.asarray(ref_trajectory[:N], dtype=float)
        x0 = np.asarray(x0, dtype=float)

        bounds = []
        for _ in range(N):
            bounds.append(
                (-self.max_steer_rad, self.max_steer_rad)
            )
            bounds.append(
                (-self.k_a, self.k_a)
            )

        def objective(u):
            state = x0.copy()
            prev_delta = current_steer
            cost = 0.0

            for k in range(N):
                delta = u[2 * k]
                accel = u[2 * k + 1]

                x = state[0]
                y = state[1]
                yaw = state[2]
                v = state[3]

                x_ref = ref[k, 0]
                y_ref = ref[k, 1]
                yaw_ref = ref[k, 2]
                v_ref = ref[k, 3]

                dx = x - x_ref
                dy = y - y_ref

                c = math.cos(yaw_ref)
                s = math.sin(yaw_ref)

                cte = -s * dx + c * dy

                yaw_error = (yaw - yaw_ref + math.pi) % (2.0 * math.pi) - math.pi
                speed_error = v - v_ref

                cost += self.w_lat * cte ** 2
                cost += self.w_yaw * yaw_error ** 2
                cost += self.w_v * speed_error ** 2
                cost += self.w_steer * delta ** 2
                cost += self.w_dsteer * (delta - prev_delta) ** 2
                cost += self.w_accel * accel ** 2

                state[0] = x + v * math.cos(yaw) * self.dt
                state[1] = y + v * math.sin(yaw) * self.dt
                state[2] = yaw + (v / self.L) * math.tan(delta) * self.dt
                state[3] = max(0.0, v + accel * self.dt)

                state[2] = (
                    state[2] + math.pi
                ) % (2.0 * math.pi) - math.pi

                prev_delta = delta

            return float(cost)

        if len(self.last_u) >= 2 * N:
            u_init = self.last_u[:2 * N].copy()

            if N > 1:
                u_init[:-2] = u_init[2:]
                u_init[-2:] = u_init[-4:-2]
        else:
            u_init = np.zeros(2 * N)

        u_init = np.clip(
            u_init,
            [b[0] for b in bounds],
            [b[1] for b in bounds]
        )

        result = minimize(
            objective,
            u_init,
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 25, 'ftol': 1e-3}
        )

        if result.success:
            optimal_u = result.x
        else:
            optimal_u = u_init

        self.last_u = np.zeros(2 * self.N)
        self.last_u[:2 * N] = optimal_u

        delta_cmd = float(optimal_u[0])
        accel_cmd = float(optimal_u[1])

        throttle_cmd = float(
            np.clip(
                accel_cmd / self.k_a,
                -1.0,
                1.0
            )
        )

        return delta_cmd, throttle_cmd
