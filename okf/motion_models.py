from abc import ABC
from typing import Tuple, Callable

import torch
import numpy as np
from pyquaternion import Quaternion

from okf_lib.okf.utils import warp_to_pi


class EKFMotionModel(ABC):
    def x_dim(self) -> int:
        pass

    def z_dim(self) -> int:
        pass

    def f(self, state) -> torch.Tensor:
        """True state transition function"""
        pass

    def h(self, state) -> torch.Tensor:
        """True state to measurement function"""
        pass

    def jacobian_of_f(self, x_row) -> torch.Tensor:
        pass

    def jacobian_of_h(self, state) -> torch.Tensor:
        pass

    def initial_observation_to_state(self, obs_vec) -> torch.Tensor:
        """Maps first observation to state"""
        pass

    # @staticmethod
    # def initial_p() -> torch.Tensor:
    #     pass
    #
    # @staticmethod
    # def loss_fun() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    #     """
    #     Should return a callable f(pred, x) -> torch scalar loss.
    #     """
    #     pass


class CTRA(EKFMotionModel):
    def __init__(self, has_velo: bool, dt: float):
        self.has_velo = has_velo
        self.dt = dt
        self.dim_x = 10
        self.dim_z = 9 if self.has_velo else 7

    def x_dim(self) -> int:
        return self.dim_x

    def z_dim(self) -> int:
        return self.dim_z

    def f(self, state):
        assert state.shape[0] == 10, "state vector number in CTRA must equal to 10"

        dt = self.dt
        x, y, z, w, l, h, v, a, theta, omega = state.detach().numpy()
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        next_v = v + a * dt
        next_ry = warp_to_pi(theta + omega * dt)
        a_out = 0.0

        # corner case(tiny yaw rate), prevent divide-by-zero overflow
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt**2 / 2
            predict_state = [
                x + displacement * yaw_cos,
                y + displacement * yaw_sin,
                z,
                w,
                l,
                h,
                next_v,
                a_out,
                next_ry,
                omega,
            ]
        else:
            ry_rate_inv_square = 1.0 / (omega * omega)
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)

            predict_state = [
                x
                + ry_rate_inv_square
                * (
                    next_v * omega * next_yaw_sin
                    + a * next_yaw_cos
                    - v * omega * yaw_sin
                    - a * yaw_cos
                ),
                y
                + ry_rate_inv_square
                * (
                    -next_v * omega * next_yaw_cos
                    + a * next_yaw_sin
                    + v * omega * yaw_cos
                    - a * yaw_sin
                ),
                z,
                w,
                l,
                h,
                next_v,
                a,
                next_ry,
                omega,
            ]

        return torch.tensor(predict_state)

    def h(self, state) -> torch.Tensor:
        assert state.shape[0] == 10, "state vector number in CTRA must equal to 10"

        x, y, z, w, l, h, v, _, theta, _ = state.detach().numpy()
        if self.has_velo:
            state_info = [x, y, z, w, l, h, v * np.cos(theta), v * np.sin(theta), theta]
        else:
            state_info = [x, y, z, w, l, h, theta]

        return torch.tensor(state_info)

    def jacobian_of_f(self, x_row) -> torch.Tensor:
        dt = self.dt
        _, _, _, _, _, _, v, a, theta, omega = x_row.detach().numpy()
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)

        # corner case, tiny turn rate
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt**2 / 2
            F = torch.tensor(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        dt * yaw_cos,
                        dt**2 * yaw_cos / 2,
                        -displacement * yaw_sin,
                        0,
                    ],
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        dt * yaw_sin,
                        dt**2 * yaw_sin / 2,
                        displacement * yaw_cos,
                        0,
                    ],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
        else:
            ry_rate_inv, ry_rate_inv_square, ry_rate_inv_cube = (
                1 / omega,
                1 / (omega * omega),
                1 / (omega * omega * omega),
            )
            next_v, next_ry = v + a * dt, theta + omega * dt
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            F = torch.tensor(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -ry_rate_inv * (yaw_sin - next_yaw_sin),
                        -ry_rate_inv_square * (yaw_cos - next_yaw_cos)
                        + ry_rate_inv * dt * next_yaw_sin,
                        ry_rate_inv_square * a * (yaw_sin - next_yaw_sin)
                        + ry_rate_inv * (next_v * next_yaw_cos - v * yaw_cos),
                        ry_rate_inv_cube * 2 * a * (yaw_cos - next_yaw_cos)
                        + ry_rate_inv_square
                        * (v * yaw_sin - v * next_yaw_sin - 2 * a * dt * next_yaw_sin)
                        + ry_rate_inv * dt * next_v * next_yaw_cos,
                    ],
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        ry_rate_inv * (yaw_cos - next_yaw_cos),
                        -ry_rate_inv_square * (yaw_sin - next_yaw_sin)
                        - ry_rate_inv * dt * next_yaw_cos,
                        ry_rate_inv_square * a * (-yaw_cos + next_yaw_cos)
                        + ry_rate_inv * (next_v * next_yaw_sin - v * yaw_sin),
                        ry_rate_inv_cube * 2 * a * (yaw_sin - next_yaw_sin)
                        + ry_rate_inv_square
                        * (v * next_yaw_cos - v * yaw_cos + 2 * a * dt * next_yaw_cos)
                        + ry_rate_inv * dt * next_v * next_yaw_sin,
                    ],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )

        return F

    def jacobian_of_h(self, state) -> torch.Tensor:
        if self.has_velo:
            _, _, _, _, _, _, v, _, theta, _ = state
            yaw_sin, yaw_cos = torch.sin(theta), torch.cos(theta)
            H = torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, yaw_cos, 0, -v * yaw_sin, 0],
                    [0, 0, 0, 0, 0, 0, yaw_sin, 0, v * yaw_cos, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                ]
            )
        else:
            H = torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                ]
            )

        return H.to(torch.float64)

    def initial_observation_to_state(self, obs_vec):
        if self.has_velo:
            x, y, z, w, l, h, vx, vy, yaw = obs_vec
            v = torch.sqrt(vx**2 + vy**2)
            return torch.tensor([x, y, z, w, l, h, v, 0, yaw, 0])
        else:
            x, y, z, w, l, h, yaw = obs_vec
            return torch.tensor([x, y, z, w, l, h, 0, 0, yaw, 0])

    @staticmethod
    def initial_p() -> torch.Tensor:
        return torch.diag(
            torch.tensor([4, 4, 4, 4, 4, 4, 1000, 4, 1, 0.1], dtype=torch.float64)
        )

    @staticmethod
    def loss_fun():
        def _loss(pred, x):
            # indices: 0..5 = x,y,z,w,l,h ; 8 = yaw
            pos_loss = (pred[:6] - x[:6]) ** 2
            yaw_loss = (pred[8] - x[8]) ** 2
            return pos_loss.sum() + yaw_loss.sum()

        return _loss


class Bicycle(EKFMotionModel):
    def __init__(self, has_velo: bool, dt: float) -> None:
        super().__init__()
        self.has_velo, self.dt, self.SD = has_velo, dt, 10
        self.MD = 9 if self.has_velo else 7
        self.w_r, self.lf_r = 0.8, 0.5

    def x_dim(self) -> int:
        return self.SD

    def z_dim(self) -> int:
        return self.MD

    def jacobian_of_f(self, state) -> torch.Tensor:
        dt = self.dt
        _, _, _, _, l, _, v, a, theta, sigma = state.detach().numpy()
        beta, _, lr = self.get_bic_beta(l, sigma)

        sin_yaw, cos_yaw = np.sin(theta), np.cos(theta)

        # corner case, tiny beta
        if abs(beta) < 0.001:
            displacement = a * dt**2 / 2 + dt * v
            F = torch.tensor(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        dt * cos_yaw,
                        dt**2 * cos_yaw / 2,
                        -displacement * sin_yaw,
                        0,
                    ],
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        dt * sin_yaw,
                        dt**2 * sin_yaw / 2,
                        displacement * cos_yaw,
                        0,
                    ],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
        else:
            next_yaw, sin_beta = theta + v / lr * np.sin(beta) * dt, np.sin(beta)
            v_yaw, next_v_yaw = beta + theta, beta + next_yaw
            F = torch.tensor(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        dt * np.cos(next_v_yaw),
                        0,
                        -lr * np.cos(v_yaw) / sin_beta
                        + lr * np.cos(next_v_yaw) / sin_beta,
                        0,
                    ],
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        dt * np.sin(next_v_yaw),
                        0,
                        -lr * np.sin(v_yaw) / sin_beta
                        + lr * np.sin(next_v_yaw) / sin_beta,
                        0,
                    ],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, dt / lr * sin_beta, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
        return F

    def jacobian_of_h(self, state) -> torch.Tensor:
        _, _, _, _, l, _, v, _, theta, sigma = state.detach().numpy()

        geo2gra_dist, lr2l = self.gra_to_geo_dist(l), self.w_r * (0.5 - self.lf_r)
        sin_yaw, cos_yaw = np.sin(theta), np.cos(theta)

        if self.has_velo:
            beta, _, _ = self.get_bic_beta(l, sigma)
            v_yaw = beta + theta
            sin_v_yaw, cos_v_yaw = np.sin(v_yaw), np.cos(v_yaw)
            H = torch.tensor(
                [
                    [1, 0, 0, 0, -lr2l * cos_yaw, 0, 0, 0, geo2gra_dist * sin_yaw, 0],
                    [0, 1, 0, 0, -lr2l * sin_yaw, 0, 0, 0, -geo2gra_dist * cos_yaw, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, cos_v_yaw, 0, -v * sin_v_yaw, 0],
                    [0, 0, 0, 0, 0, 0, sin_v_yaw, 0, v * cos_v_yaw, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                ]
            )
        else:
            H = torch.tensor(
                [
                    [1, 0, 0, 0, -lr2l * cos_yaw, 0, 0, 0, geo2gra_dist * sin_yaw, 0],
                    [0, 1, 0, 0, -lr2l * sin_yaw, 0, 0, 0, -geo2gra_dist * cos_yaw, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                ]
            )

        return H

    def f(self, state) -> torch.Tensor:
        assert state.shape[0] == 10, "state vector number in Bicycle must equal to 10"

        dt = self.dt
        x_gra, y_gra, z, w, l, h, v, a, theta, sigma = state.detach().numpy()  # TODO
        beta, _, lr = self.get_bic_beta(l, sigma)

        # corner case, tiny yaw rate
        if abs(beta) > 0.001:
            next_yaw = theta + v / lr * np.sin(beta) * dt
            v_yaw, next_v_yaw = beta + theta, beta + next_yaw
            predict_state = torch.tensor(
                [
                    x_gra
                    + (lr * (np.sin(next_v_yaw) - np.sin(v_yaw)))
                    / np.sin(beta),  # x_gra
                    y_gra
                    - (lr * (np.cos(next_v_yaw) - np.cos(v_yaw)))
                    / np.sin(beta),  # y_gra
                    z,
                    w,
                    l,
                    h,
                    v,
                    0,
                    next_yaw,
                    sigma,
                ]
            )
        else:
            displacement = v * dt + a * dt**2 / 2
            predict_state = torch.tensor(
                [
                    x_gra + displacement * np.cos(theta),
                    y_gra + displacement * np.sin(theta),
                    z,
                    w,
                    l,
                    h,
                    v + a * dt,
                    a,
                    theta,
                    sigma,
                ]
            )
        return predict_state

    def h(self, state) -> torch.Tensor:
        assert state.shape[0] == 10, "state vector number in BICYCLE must equal to 10"

        x_gra, y_gra, z, w, l, h, v, _, theta, sigma = state.detach().numpy()

        beta, _, _ = self.get_bic_beta(l, sigma)
        geo2gra_dist = self.gra_to_geo_dist(l)

        if self.has_velo:
            meas_state = [
                x_gra - geo2gra_dist * np.cos(theta),
                y_gra - geo2gra_dist * np.sin(theta),
                z,
                w,
                l,
                h,
                v * np.cos(theta + beta),
                v * np.sin(theta + beta),
                theta,
            ]
        else:
            meas_state = [
                x_gra - geo2gra_dist * np.cos(theta),
                y_gra - geo2gra_dist * np.sin(theta),
                z,
                w,
                l,
                h,
                theta,
            ]

        return torch.tensor(meas_state)

    def get_bic_beta(self, length: float, sigma: float) -> Tuple[float, float, float]:
        """get the angle between the object velocity and the X-axis of the coordinate system

        Args:
            length (float): object length
            sigma (float): the steering angle, radians

        Returns:
            beta (float): the angle between the object velocity and X-axis, radians
            lf (float): distance from CG to front axle
            lr (float): distance from CG to rear axle
        """

        # distances still depend on length
        lf = float(length) * self.w_r * self.lf_r
        lr = float(length) * self.w_r * (1.0 - self.lf_r)

        # geometric ratio lr / (lr + lf) simplifies to (1 - lf_r)
        k = 1.0 - self.lf_r  # independent of length and w_r

        # handle bad / extreme sigma
        if not np.isfinite(sigma):
            beta = 0.0
        else:
            # wrap sigma into [-pi, pi]
            sigma_wrapped = (float(sigma) + np.pi) % (2.0 * np.pi) - np.pi

            # clamp to avoid tan() blowing up exactly at ±pi/2
            max_steer = np.deg2rad(89.0)  # physically also reasonable
            sigma_limited = np.clip(sigma_wrapped, -max_steer, max_steer)

            beta = float(np.arctan(k * np.tan(sigma_limited)))

        return beta, lf, lr

    def gra_to_geo_dist(self, length: float) -> float:
        """get gra center to geo center distance

        Args:
            length (float): object length

        Returns:
            float: gra center to geo center distance
        """
        return length * self.w_r * (0.5 - self.lf_r)

    def initial_observation_to_state(self, obs_vec):
        """
        Map first measurement (geometric center) to Bicycle state (gravity center).

        obs_vec (has_velo=True):
            [x_geo, y_geo, z, w, l, h, vx, vy, yaw]

        obs_vec (has_velo=False):
            [x_geo, y_geo, z, w, l, h, yaw]

        state:
            [x_gra, y_gra, z, w, l, h, v, a, theta, sigma]
        """
        if self.has_velo:
            x_geo, y_geo, z, w, l, h, vx, vy, yaw = obs_vec

            # --- heading / speed from obs ---
            theta = float(yaw)
            v = float(np.hypot(vx, vy))

            # --- geo -> gra (invert h()) ---
            # In h(): x_geo = x_gra - d * cos(theta), so x_gra = x_geo + d * cos(theta)
            geo2gra_dist = self.gra_to_geo_dist(float(l))
            x_gra = float(x_geo) + geo2gra_dist * np.cos(theta)
            y_gra = float(y_geo) + geo2gra_dist * np.sin(theta)

            # --- estimate steering angle sigma from velocity direction ---
            # velocity direction in world frame
            if v > 1e-3:
                vel_dir = float(np.arctan2(vy, vx))  # direction of (vx, vy)

                # slip angle beta = vel_dir - theta (wrap to [-pi, pi])
                beta = vel_dir - theta
                beta = (beta + np.pi) % (2 * np.pi) - np.pi

                # bicycle geometry
                lf = float(l) * self.w_r * self.lf_r
                lr = float(l) * self.w_r * (1.0 - self.lf_r)
                k = lr / (lr + lf + 1e-8)

                # from get_bic_beta: tan(beta) = k * tan(sigma)  =>  sigma = atan( tan(beta) / k )
                sigma = float(np.arctan(np.tan(beta) / (k + 1e-8)))
            else:
                # If no meaningful velocity, just start straight
                sigma = 0.0

            a = 0.0  # no info about acceleration from a single obs

            return torch.tensor(
                [
                    x_gra,
                    y_gra,
                    float(z),
                    float(w),
                    float(l),
                    float(h),
                    v,
                    a,
                    theta,
                    sigma,
                ],
                dtype=torch.float32,
            )
        else:
            # No velocity: we can only set position, size, heading and assume v=a=sigma=0
            x_geo, y_geo, z, w, l, h, yaw = obs_vec
            theta = float(yaw)

            geo2gra_dist = self.gra_to_geo_dist(float(l))
            x_gra = float(x_geo) + geo2gra_dist * np.cos(theta)
            y_gra = float(y_geo) + geo2gra_dist * np.sin(theta)

            return torch.tensor(
                [
                    x_gra,
                    y_gra,
                    float(z),
                    float(w),
                    float(l),
                    float(h),
                    0.0,
                    0.0,
                    theta,
                    0.0,
                ],
                dtype=torch.float32,
            )

    @staticmethod
    def loss_fun():
        def _loss(pred, x):
            # indices: 0..5 = x,y,z,w,l,h ; 8 = ry
            pos_loss = (pred[:6] - x[:6]) ** 2
            yaw_loss = (pred[8] - x[8]) ** 2
            return pos_loss.sum() + yaw_loss.sum()

        return _loss
