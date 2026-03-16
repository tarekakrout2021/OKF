"""
This module implements the Optimized Kalman Filter (OKF) class.
The OKF is similar to any KF implementation, but the parameters are torch variables such that they can be optimized.

The OKF class includes:
- Interfaces:
    - A constructor (see documentation in __init__()).
    - init_state(): Initialize the model before a new sequence of observations.
    - reset_model(): Reset the model parameters (Q,R).
    - save_model(), load_model().
- Methods to apply the model (to be run iteratively):
    - predict(): Advance a single time-step and update the state.
    - update(): Process a new observation and update the state.
- Tuning the model parameters:
    - estimate_noise(): Set the parameters to be the sample covariance matrices of the noise in the given data.
    - Optimization of the parameters wrt a loss function is implemented in a separate module.
- Methods to observe the model: get_Q(), get_R(), display_params().

Written by Ido Greenberg, 2021
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch import matmul as mp

from . import utils
from .motion_models import EKFMotionModel, CTRA


class ObservationNoiseMode(Enum):
    LEARNED = "learned"
    PROVIDED = "provided"


class ObservationNoiseStrategy(nn.Module, ABC):
    def __init__(self, dim_z: int, optimize: bool):
        super().__init__()
        self.dim_z = dim_z
        self.optimize = optimize

    @property
    @abstractmethod
    def is_learned(self) -> bool:
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def covariance(self, r=None) -> torch.Tensor:
        pass

    @abstractmethod
    def estimate_from_residuals(self, delta: np.ndarray):
        pass

    @abstractmethod
    def get_covariance(self, to_numpy: bool = True):
        pass

    @abstractmethod
    def export_state(self):
        pass

    @abstractmethod
    def load_exported_state(self, state):
        pass


class LearnedObservationNoise(ObservationNoiseStrategy):
    def __init__(self, dim_z: int, optimize: bool, R0=1):
        super().__init__(dim_z=dim_z, optimize=optimize)
        self.R0 = R0
        self.R_D = None
        self.R_L = None
        self.reset_parameters()

    @property
    def is_learned(self) -> bool:
        return True

    def reset_parameters(self):
        if isinstance(self.R0, torch.Tensor) and len(self.R0.shape):
            R_D, R_L = OKF.encode_SPD(self.R0)
        else:
            R_D = (self.R0 * (0.5 + torch.rand(self.dim_z, dtype=torch.double))).log()
            R_L = (
                self.R0
                / 5
                * torch.randn(self.dim_z * (self.dim_z - 1) // 2, dtype=torch.double)
            )

        if self.optimize:
            self.R_D = nn.Parameter(R_D, requires_grad=True)
            self.R_L = nn.Parameter(R_L, requires_grad=True)
        else:
            self.R_D = R_D
            self.R_L = R_L

    def covariance(self, r=None) -> torch.Tensor:
        return OKF.get_SPD(self.R_D, self.R_L)

    def estimate_from_residuals(self, delta: np.ndarray):
        R = torch.tensor(np.cov(delta.T), dtype=torch.double)
        R_D, R_L = OKF.encode_SPD(R)
        if self.optimize:
            with torch.no_grad():
                self.R_D.copy_(R_D)
                self.R_L.copy_(R_L)
        else:
            self.R_D, self.R_L = R_D, R_L

    def get_covariance(self, to_numpy: bool = True):
        A = self.covariance()
        if to_numpy:
            A = A.detach().numpy()
        return A

    def export_state(self):
        return self.R_D, self.R_L

    def load_exported_state(self, state):
        R_D, R_L = state
        if self.optimize:
            with torch.no_grad():
                self.R_D.copy_(R_D)
                self.R_L.copy_(R_L)
        else:
            self.R_D, self.R_L = R_D, R_L


class ProvidedDiagonalObservationNoise(ObservationNoiseStrategy):
    def __init__(self, dim_z: int, optimize: bool):
        super().__init__(dim_z=dim_z, optimize=optimize)

    @property
    def is_learned(self) -> bool:
        return False

    def reset_parameters(self):
        return None

    def covariance(self, r=None) -> torch.Tensor:
        if r is None:
            raise ValueError("Measurement-provided observation noise requires per-step uncertainty 'r'.")
        if not torch.is_tensor(r):
            r = torch.tensor(r, dtype=torch.double)
        return torch.diag(r)

    def estimate_from_residuals(self, delta: np.ndarray):
        return None

    def get_covariance(self, to_numpy: bool = True):
        raise ValueError("Observation covariance is provided per measurement and is not stored in the model.")

    def export_state(self):
        return None

    def load_exported_state(self, state):
        return None


class OKF(nn.Module):
    def __init__(
        self,
        motion_model: EKFMotionModel,
        model_name="OKF",
        P0=1e3,
        Q0=1,
        R0=1,
        x0=None,
        optimize=True,
        model_files_path="models/",
        observation_noise_mode: ObservationNoiseMode = ObservationNoiseMode.LEARNED,
        observation_noise_strategy: ObservationNoiseStrategy = None,
    ):
        """
        A model of KF whose parameters (Q,R) are pytorch tensors and can be optimized wrt a loss function.

        Under the KF assumptions, such optimization (wrt the MSE of the model-predictions) is equivalent to simply
        calculate the sample covariance matrices of the noise (given corresponding data). However, the KF assumptions
        do not usually hold in practical problems - which is sometimes goes unnoticed. Optimization can obtain better
        accuracy in such cases.

        :param motion_model: e.g CTRA, Bicycle.
        :param model_name: Model name [str].
        :param P0: The initial value of the uncertainty matrix P, used to initialize P every new trajectory. If scalar,
                   the initial P is P0*eye(dim_x) [numeric OR pytorch tensor with type double and shape (dim_x,dim_x);
                   default=1e3].
        :param Q0: The initial value of the process-noise covariance matrix Q, from which the optimization begins.
                   If scalar, it is used as a scale drawing the initial matrix randomly [positive numeric OR pytorch
                   tensor with type double and shape (dim_x,dim_x); default=1; only used if optimize==True].
        :param R0: The initial value of the observation-noise covariance matrix R, from which the optimization begins.
                   If scalar, it is used as a scale drawing the initial matrix randomly [positive numeric OR pytorch
                   tensor with type double and shape (dim_z,dim_z); default=1; only used if optimize==True].
        :param x0: The initial value of the state x. If None, then the state will only be initialized after the first
                   observation z, as x=init_z2x(z). If scalar, x=x0*ones(dim_x) is used [scalar OR pytorch tensor with
                   type double and shape dim_x OR None; default=None].
        :param optimize: Whether to tune the parameters Q,R by optimization or using the standard sample covariance
                         matrices of the noise.
        :param model_files_path: Directory path to save the model in [str].
        :param observation_noise_mode: Whether R is learned globally or provided per measurement.
        :param observation_noise_strategy: Optional custom strategy for observation noise.
        """
        nn.Module.__init__(self)
        self.model_name = model_name
        self.base_path = model_files_path
        self.optimize = optimize
        self.motion_model = motion_model
        self.dim_x = motion_model.x_dim()
        self.dim_z = motion_model.z_dim()
        self.true_fun = motion_model.f

        self.F = motion_model.jacobian_of_f
        self.H = motion_model.jacobian_of_h
        self.is_H_fun = callable(self.H)
        self.is_F_fun = callable(self.true_fun)
        self.state_to_measure = motion_model.h

        if x0 is None:
            x0 = self.dim_x * [None]
        elif not torch.is_tensor(x0):
            x0 = x0 * torch.ones(self.dim_x, dtype=torch.double)
        self.x0 = x0

        self.P0 = (
            motion_model.initial_p()
            if isinstance(self.motion_model, CTRA)
            else P0 * torch.eye(self.dim_x, dtype=torch.double)
        )
        self.Q0 = Q0
        self.R0 = R0
        self.Q_D, self.Q_L = 2 * [None]
        self.observation_noise = self._build_observation_noise_strategy(
            observation_noise_mode=observation_noise_mode,
            observation_noise_strategy=observation_noise_strategy,
            R0=R0,
        )
        self.reset_model()

        self.z2x = motion_model.initial_observation_to_state

        self.loss_fun = motion_model.loss_fun()
        if self.loss_fun is None:
            self.loss_fun = lambda pred, x: ((pred - x) ** 2).sum()

        if len(self.x0) != self.dim_x:
            raise ValueError(
                f"Bad input dimension: len(x0) = {len(self.x0)} != {self.dim_x}."
            )
        if self.P0.shape != (self.dim_x, self.dim_x):
            raise ValueError(
                f"Bad input dimension: P0.shape = {self.P0.shape} != {(self.dim_x, self.dim_x)}."
            )

        self.x = None
        self.z = None
        self.P = None
        self.init_state()

        self.K_history = []

    @property
    def train_R(self) -> bool:
        return self.observation_noise.is_learned

    def _build_observation_noise_strategy(
        self,
        observation_noise_mode: ObservationNoiseMode,
        observation_noise_strategy: ObservationNoiseStrategy,
        R0,
    ) -> ObservationNoiseStrategy:
        if observation_noise_strategy is not None:
            return observation_noise_strategy
        if observation_noise_mode == ObservationNoiseMode.LEARNED:
            return LearnedObservationNoise(dim_z=self.dim_z, optimize=self.optimize, R0=R0)
        if observation_noise_mode == ObservationNoiseMode.PROVIDED:
            return ProvidedDiagonalObservationNoise(dim_z=self.dim_z, optimize=self.optimize)
        raise ValueError(f"Unsupported observation noise mode: {observation_noise_mode}")

    def init_state(self):
        """Initialize the estimate (x,P) and the observation (z) before a new sequence of observations (trajectory)."""
        self.x = self.x0
        self.z = self.dim_z * [None]
        self.P = self.P0

    def reset_model(self):
        """Reset the model parameters (Q,R)."""
        if isinstance(self.Q0, torch.Tensor) and len(self.Q0.shape):
            Q_D, Q_L = OKF.encode_SPD(self.Q0)
        else:
            Q_D = (self.Q0 * (0.5 + torch.rand(self.dim_x, dtype=torch.double))).log()
            Q_L = (
                self.Q0
                / 5
                * torch.randn(self.dim_x * (self.dim_x - 1) // 2, dtype=torch.double)
            )

        if self.optimize:
            self.Q_D = nn.Parameter(Q_D, requires_grad=True)
            self.Q_L = nn.Parameter(Q_L, requires_grad=True)
        else:
            self.Q_D, self.Q_L = Q_D, Q_L

        self.observation_noise.reset_parameters()

    def save_model(self, fname=None, base_path=None, assert_suffices=True):
        fpath = self.get_model_path(fname, base_path, assert_suffices)
        if self.optimize:
            torch.save(self.state_dict(), fpath)
        else:
            torch.save(
                {
                    "Q_D": self.Q_D,
                    "Q_L": self.Q_L,
                    "observation_noise": self.observation_noise.export_state(),
                },
                fpath,
            )

    def load_model(self, fname=None, base_path=None, assert_suffices=True):
        fpath = self.get_model_path(fname, base_path, assert_suffices)
        if self.optimize:
            self.load_state_dict(torch.load(fpath))
        else:
            state = torch.load(fpath)
            self.Q_D = state["Q_D"]
            self.Q_L = state["Q_L"]
            self.observation_noise.load_exported_state(state.get("observation_noise"))

    def get_model_path(self, fname=None, base_path=None, assert_suffices=True):
        if base_path is None:
            base_path = self.base_path
        if fname is None:
            fname = self.model_name
        if assert_suffices:
            if base_path[-1] != "/":
                base_path += "/"
            if self.optimize:
                if not fname.endswith(".m"):
                    fname += ".m"
            else:
                if not fname.endswith(".noise"):
                    fname += ".noise"
        return base_path + fname

    def predict(self):
        if self.x[0] is None:
            return
        F = self.F(self.x) if self.is_F_fun else self.F
        Q = OKF.get_SPD(self.Q_D, self.Q_L)
        self.x = self.true_fun(self.x)
        utils.warpStateYawToPi(self.x)
        self.P = mp(mp(F, self.P), F.T) + Q

    def update(self, z, r):
        self.z = torch.tensor(z)
        is_x_none = False
        for x in self.x:
            if x is None:
                is_x_none = True
        if is_x_none:
            H = self.H(torch.tensor([0.0] * len(self.x), dtype=torch.double)) if self.is_H_fun else self.H
        else:
            H = self.H(self.x) if self.is_H_fun else self.H

        R = self.observation_noise.covariance(r)
        Ht = H.T
        PHt = mp(self.P, Ht)
        self.S = mp(H, PHt) + R
        K = mp(PHt, self.S.inverse())

        self.K_history.append(K.detach().cpu().numpy())

        I_KH = torch.eye(self.P.shape[0], dtype=self.P.dtype) - mp(K, H)
        self.P = mp(mp(I_KH, self.P), I_KH.T) + mp(mp(K, R), K.T)
        self.P = 0.5 * (self.P + self.P.T)

        if self.x[0] is not None:
            res = self.z - self.state_to_measure(self.x)
            utils.warpResYawToPi(res)
            self.x = self.x + mp(K, res)
            utils.warpStateYawToPi(self.x)
        else:
            self.x = self.z2x(self.z)
            utils.warpStateYawToPi(self.x)

    @staticmethod
    def get_SPD(D, L):
        n = len(D)
        A = D.exp().diag()
        ids = torch.tril_indices(n, n, -1)
        A[ids[0, :], ids[1, :]] = L
        return mp(A, A.T)

    @staticmethod
    def encode_SPD(A, eps=1e-6):
        n = A.shape[0]
        A = torch.linalg.cholesky(A + eps * torch.eye(n, dtype=A.dtype))
        D = A.diag()
        D = D.log()
        ids = torch.tril_indices(n, n, -1)
        L = A[ids[0, :], ids[1, :]]
        return D, L

    def estimate_noise(self, X, Z):
        """
        Tune the KF by noise estimation:
        Set the parameters Q,R to be the sample covariance matrices of the noise in the given data.

        :param X: a list of targets states. X[i] = numpy array of type double and shape (n_time_steps(i), dim_x).
        :param Z: a list of targets observations. Z[i] = numpy array of type double and shape (n_time_steps(i), dim_z).
        """

        X1 = torch.cat([torch.tensor(x[:-1], dtype=torch.double) for x in X], dim=0)
        X2 = torch.cat([torch.tensor(x[1:], dtype=torch.double) for x in X], dim=0)
        if self.is_F_fun:
            Fx1 = torch.stack([self.true_fun(x) for x in X1], dim=0)
        else:
            Fx1 = mp(self.F, X1.T).T
        res = Fx1 - X2
        for i in range(res.shape[0]):
            utils.warpStateYawToPi(res[i])

        Q = torch.tensor(np.cov(res.T.detach().numpy()), dtype=torch.double)

        H = []
        Z = np.concatenate(Z, axis=0)
        if self.is_H_fun:
            for x in X:
                for x_t in x:
                    H.append(self.H(torch.tensor(x_t, dtype=torch.double)))
        else:
            H = len(X) * [self.H]

        Hx = np.concatenate(
            [mp(h, torch.tensor(x, dtype=torch.double).T).T.detach().numpy() for x, h in zip(X, H)], axis=0
        )

        delta = Z - Hx
        for i in range(delta.shape[0]):
            utils.warpResYawToPi(delta[i])

        Q_D, Q_L = OKF.encode_SPD(Q)
        if self.optimize:
            with torch.no_grad():
                self.Q_D.copy_(Q_D)
                self.Q_L.copy_(Q_L)
        else:
            self.Q_D, self.Q_L = Q_D, Q_L

        self.observation_noise.estimate_from_residuals(delta)

    def get_Q(self, to_numpy=True):
        A = OKF.get_SPD(self.Q_D, self.Q_L)
        if to_numpy:
            A = A.detach().numpy()
        return A

    def get_R(self, to_numpy=True):
        return self.observation_noise.get_covariance(to_numpy=to_numpy)

    def display_params(self, n_digits=0, fontsize=15, axsize=(4.5, 3.5)):
        matrices = [("Q", self.get_Q())]
        if self.train_R:
            matrices.append(("R", self.get_R()))

        axs = utils.Axes(1, len(matrices), axsize=axsize)
        if len(matrices) == 1:
            axs = [axs[0]]
        for i, (name, A) in enumerate(matrices):
            h = sns.heatmap(
                A,
                annot=True,
                fmt=f".{n_digits:d}f",
                cmap="Reds",
                ax=axs[i],
                annot_kws=None if fontsize is None else dict(size=fontsize),
            )
            h.xaxis.set_ticks_position("top")
            axs[i].set_title(f"[{self.model_name}] {name}", fontsize=fontsize + 2)
        plt.tight_layout()
        return axs
