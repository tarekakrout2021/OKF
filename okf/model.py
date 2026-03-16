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

import types, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch import matmul as mp

from . import utils
from .motion_models import EKFMotionModel, CTRA


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
        train_R=True
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
        :param init_z2x: A function that receives the first observation and returns the first estimate of the state.
                         If array instead of a callable, just initializing the state to the given array.
                         [fun(z); default=identity function; if dim_x!=dim_z, another function must be specified].
        :param loss_fun: Loss function to optimize [fun(predicted_x, true_x); default=MSE; only used if optimize==True].
        :param optimize: Whether to tune the parameters Q,R by optimization or using the standard sample covariance
                         matrices of the noise.
        :param model_files_path: Directory path to save the model in [str].
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

        # if not torch.is_tensor(P0):
        #     P0 = P0 * torch.eye(self.dim_x, dtype=torch.double)
        self.P0 = (
            motion_model.initial_p()
            if isinstance(self.motion_model, CTRA)
            else P0 * torch.eye(self.dim_x, dtype=torch.double)
        )
        self.Q0 = Q0
        self.train_R = train_R
        self.R0 = R0  # todo
        self.Q_D, self.Q_L, self.R_D, self.R_L = 4 * [None]
        self.reset_model()

        self.z2x = motion_model.initial_observation_to_state

        self.loss_fun = motion_model.loss_fun()
        if self.loss_fun is None:
            self.loss_fun = lambda pred, x: ((pred - x) ** 2).sum()

        # verify dimensions
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

        self.K_history = []  # stores Kalman Gain

    def init_state(self):
        """Initialize the estimate (x,P) and the observation (z) before a new sequence of observations (trajectory)."""
        self.x = self.x0
        self.z = self.dim_z * [None]
        self.P = self.P0

    def reset_model(self):
        """Reset the model parameters (Q,R)."""
        # Q
        if isinstance(self.Q0, torch.Tensor) and len(self.Q0.shape):
            # given as a 2D tensor
            Q_D, Q_L = OKF.encode_SPD(self.Q0)
        else:
            # given as a scale for randomization
            Q_D = (self.Q0 * (0.5 + torch.rand(self.dim_x, dtype=torch.double))).log()
            Q_L = (
                self.Q0
                / 5
                * torch.randn(self.dim_x * (self.dim_x - 1) // 2, dtype=torch.double)
            )
        # R
        if self.train_R:
            if isinstance(self.R0, torch.Tensor) and len(self.R0.shape):
                # given as a 2D tensor
                R_D, R_L = OKF.encode_SPD(self.R0)
            else:
                # given as a scale for randomization
                R_D = (self.R0 * (0.5 + torch.rand(self.dim_z, dtype=torch.double))).log()
                R_L = (
                    self.R0
                    / 5
                    * torch.randn(self.dim_z * (self.dim_z - 1) // 2, dtype=torch.double)
                )

        if self.optimize:
            self.Q_D = nn.Parameter(Q_D, requires_grad=True)
            self.Q_L = nn.Parameter(Q_L, requires_grad=True)
            if self.train_R:
                self.R_D = nn.Parameter(R_D, requires_grad=True)
                self.R_L = nn.Parameter(R_L, requires_grad=True)
        else:
            self.Q_D, self.Q_L  = Q_D, Q_L
            if self.train_R:
                self.R_D, self.R_L =  R_D, R_L

    def save_model(self, fname=None, base_path=None, assert_suffices=True):
        fpath = self.get_model_path(fname, base_path, assert_suffices)
        if self.optimize:
            # module state dict
            torch.save(self.state_dict(), fpath)
        else:
            # save tensors
            torch.save((self.Q_D, self.Q_L, self.R_D, self.R_L), fpath)  # todo

    def load_model(self, fname=None, base_path=None, assert_suffices=True):
        fpath = self.get_model_path(fname, base_path, assert_suffices)
        if self.optimize:
            # module state dict
            self.load_state_dict(torch.load(fpath))
        else:
            # saved tensors
            self.Q_D, self.Q_L, self.R_D, self.R_L = torch.load(fpath)

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
            # state has not yet been initialized (probably no observation has yet been processed)
            return
        F = self.F(self.x) if self.is_F_fun else self.F
        Q = OKF.get_SPD(self.Q_D, self.Q_L)
        self.x = self.true_fun(self.x)
        utils.warpStateYawToPi(self.x)
        self.P = mp(mp(F, self.P), F.T) + Q

    def update(self, z, r):
        # get observation
        self.z = torch.tensor(z)
        is_x_none = False
        for x in self.x:
            if x is None:
                is_x_none = True
        if is_x_none:
            H = self.H(torch.tensor([0.0] * len(self.x))) if self.is_H_fun else self.H
        else:
            H = self.H(self.x) if self.is_H_fun else self.H
        # get update operators
        if self.train_R:
            R = OKF.get_SPD(self.R_D, self.R_L)
        else:
            R = torch.diag(r)
        Ht = H.T
        PHt = mp(self.P, Ht)
        self.S = mp(H, PHt) + R
        K = mp(PHt, self.S.inverse())

        self.K_history.append(K.detach().cpu().numpy())

        # update P
        I_KH = torch.eye(self.P.shape[0]) - mp(K, H)
        self.P = mp(mp(I_KH, self.P), I_KH.T) + mp(
            mp(K, R), K.T
        )  # equivalent to the standart formula
        # but more numerically-stable
        self.P = 0.5 * (self.P + self.P.T)  # force P to be symmetric

        # update x
        if self.x[0] is not None:
            res = self.z - self.state_to_measure(self.x)
            utils.warpResYawToPi(res)
            self.x = self.x + mp(K, res)
            utils.warpStateYawToPi(self.x)
        else:
            # utils.warpResYawToPi(self.z)
            self.x = self.z2x(self.z)
            utils.warpStateYawToPi(self.x)

    @staticmethod
    def get_SPD(D, L):
        """Convert log-diagonal entries [n] and below [n*(n-1)/2] into a SPD matrix [n^2]."""
        n = len(D)
        A = D.exp().diag()  # fill diagonal
        ids = torch.tril_indices(n, n, -1)
        A[ids[0, :], ids[1, :]] = L  # fill below-diagonal
        return mp(A, A.T)

    @staticmethod
    def encode_SPD(A, eps=1e-6):
        """Apply Cholesky decomposition to A and return the log-diagonal entires [n] and below [n*(n-1)/2]."""
        n = A.shape[0]
        A = torch.linalg.cholesky(A + eps * torch.eye(n))
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

        # Q = Cov[F*x_t - x_{t+1}]
        X1 = torch.cat([torch.tensor(x[:-1]) for x in X], dim=0)  # x_t
        X2 = torch.cat([torch.tensor(x[1:]) for x in X], dim=0)  # x_{t+1}
        if self.is_F_fun:
            Fx1 = torch.stack([self.true_fun(x) for x in X1], dim=0)
        else:
            Fx1 = mp(self.F, X1.T).T  # F*x_t # TODO: linear case
        res = Fx1 - X2
        for i in range(res.shape[0]):
            utils.warpStateYawToPi(res[i])

        # Q = Cov[F*x_t - x_{t+1}]
        Q = torch.tensor(np.cov(res.T.detach().numpy()))

        # R = Cov[z_t - H*x_t]
        H = []
        Z = np.concatenate(Z, axis=0)
        if self.is_H_fun:
            for x in X:
                for x_t in x:
                    H.append(self.H(torch.tensor(x_t)))
            # H is a list of tensors
        else:
            H = len(X) * [self.H]

        Hx = np.concatenate(
            [mp(h, torch.tensor(x).T).T.detach().numpy() for x, h in zip(X, H)], axis=0
        )
        # X3 = torch.cat([torch.tensor(x) for x in X], dim=0)
        # Hx = torch.stack([self.H(x) for x in X3], dim=0)

        delta = Z - Hx
        for i in range(delta.shape[0]):
            utils.warpResYawToPi(delta[i])

        # Cholesky parameterization
        Q_D, Q_L = OKF.encode_SPD(Q)
        if self.optimize:
            with torch.no_grad():
                self.Q_D.copy_(Q_D)
                self.Q_L.copy_(Q_L)
                if self.train_R:
                    R = torch.tensor(np.cov((delta).T))
                    R_D, R_L = OKF.encode_SPD(R)
                    self.R_D.copy_(R_D)
                    self.R_L.copy_(R_L)
        else:
            self.Q_D, self.Q_L = Q_D, Q_L
            if self.train_R:
                R = torch.tensor(np.cov((delta).T))
                R_D, R_L = OKF.encode_SPD(R)
                self.R_D, self.R_L = R_D, R_L

    def get_Q(self, to_numpy=True):
        A = OKF.get_SPD(self.Q_D, self.Q_L)
        if to_numpy:
            A = A.detach().numpy()
        return A

    def get_R(self, to_numpy=True):
        assert self.train_R == True, "R is not trained in this model."
        A = OKF.get_SPD(self.R_D, self.R_L)
        if to_numpy:
            A = A.detach().numpy()
        return A

    def display_params(self, n_digits=0, fontsize=15, axsize=(4.5, 3.5)):
        axs = utils.Axes(2, 2, axsize=axsize)
        for i, A in enumerate([self.get_Q(), self.get_R()]):
            h = sns.heatmap(
                A,
                annot=True,
                fmt=f".{n_digits:d}f",
                cmap="Reds",
                ax=axs[i],
                annot_kws=None if fontsize is None else dict(size=fontsize),
            )
            h.xaxis.set_ticks_position("top")
        axs.labs(0, title=f"[{self.model_name}] Q", fontsize=fontsize + 2)
        axs.labs(1, title=f"[{self.model_name}] R", fontsize=fontsize + 2)
        plt.tight_layout()
        return axs
