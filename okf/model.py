from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mp

from . import utils
from .motion_models import CTRA, EKFMotionModel


def _as_double_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(dtype=torch.double)
    return torch.tensor(x, dtype=torch.double)


def _gaussian_nll_from_cholesky(L: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """
    L:
      - (D, D) or
      - (B, D, D)

    residual:
      - (D,) or
      - (B, D)

    returns:
      - scalar if single sample
      - (B,) if batched
    """
    single = False
    if L.dim() == 2:
        L = L.unsqueeze(0)
        residual = residual.unsqueeze(0)
        single = True

    y = torch.linalg.solve_triangular(
        L,
        residual.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    mahal = (y * y).sum(dim=-1)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    nll = 0.5 * mahal + 0.5 * logdet
    return nll.squeeze(0) if single else nll


def _covariance_from_cholesky(L: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if L.dim() == 2:
        I = eps * torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
        return L @ L.T + I

    I = eps * torch.eye(L.shape[-1], dtype=L.dtype, device=L.device).unsqueeze(0)
    return L @ L.transpose(-1, -2) + I


class NoiseStrategyMode(Enum):
    STATIC_LEARNED = "static_learned"
    STATIC_Q_PROVIDED_R = "static_q_provided_r"
    NEURAL = "neural"


class CholeskyHead(nn.Module):
    def __init__(self, target_dim: int, diag_eps: float = 1e-4):
        super().__init__()
        self.target_dim = target_dim
        self.diag_eps = diag_eps
        self.cholesky_size = target_dim * (target_dim + 1) // 2

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """
        raw:
          - (cholesky_size,)
          - (B, cholesky_size)

        returns:
          - (target_dim, target_dim)
          - (B, target_dim, target_dim)
        """
        squeeze = False
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
            squeeze = True

        B = raw.shape[0]
        L = torch.zeros(
            (B, self.target_dim, self.target_dim),
            dtype=raw.dtype,
            device=raw.device,
        )

        idx = 0
        for i in range(self.target_dim):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = F.softplus(raw[:, idx]) + self.diag_eps
                else:
                    L[:, i, j] = raw[:, idx]
                idx += 1

        return L.squeeze(0) if squeeze else L


class CovarianceMLP(nn.Module):
    """
    Backbone MLP + linear output + dedicated Cholesky head.

    forward_raw(x) returns flattened Cholesky parameters.
    forward(x) returns the lower-triangular Cholesky matrix.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden: int = 64,
        activation: str = "silu",
        diag_eps: float = 1e-4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.hidden = hidden
        self.activation = activation
        self.diag_eps = diag_eps

        act = nn.SiLU if activation.lower() == "silu" else nn.GELU

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            act(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            act(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            act(),
        )

        self.cholesky_size = target_dim * (target_dim + 1) // 2
        self.out = nn.Linear(hidden, self.cholesky_size)
        self.cholesky_head = CholeskyHead(target_dim=target_dim, diag_eps=diag_eps)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        y = self.out(self.backbone(x))
        return y.squeeze(0) if squeeze else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cholesky_head(self.forward_raw(x))


class NoiseStrategy(nn.Module, ABC):
    def __init__(self, dim_x: int, dim_z: int, optimize: bool):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.optimize = optimize

    @property
    @abstractmethod
    def mode(self) -> NoiseStrategyMode:
        raise NotImplementedError

    @property
    def has_trainable_q(self) -> bool:
        return False

    @property
    def has_persistent_r(self) -> bool:
        return False

    @property
    def requires_measurement_uncertainty(self) -> bool:
        return False

    @abstractmethod
    def reset_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def process_covariance(self, x=None, z=None, r=None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def observation_covariance(self, x=None, z=None, r=None) -> torch.Tensor:
        raise NotImplementedError

    def estimate_from_data(self, X, Z) -> None:
        return None

    def get_process_covariance(self, to_numpy: bool = True):
        Q = self.process_covariance()
        return Q.detach().numpy() if to_numpy else Q

    def get_observation_covariance(self, to_numpy: bool = True):
        if not self.has_persistent_r:
            raise ValueError(
                "This noise strategy does not store a persistent observation covariance."
            )
        R = self.observation_covariance()
        return R.detach().numpy() if to_numpy else R

    def export_state(self):
        return None

    def load_exported_state(self, state) -> None:
        return None


class StaticNoiseStrategy(NoiseStrategy):
    def __init__(self, dim_x: int, dim_z: int, optimize: bool, Q0=1, R0=1):
        super().__init__(dim_x=dim_x, dim_z=dim_z, optimize=optimize)
        self.Q0 = Q0
        self.R0 = R0
        self.Q_D = None
        self.Q_L = None
        self.R_D = None
        self.R_L = None
        self.reset_parameters()

    @property
    def mode(self) -> NoiseStrategyMode:
        return NoiseStrategyMode.STATIC_LEARNED

    @property
    def has_trainable_q(self) -> bool:
        return True

    @property
    def has_persistent_r(self) -> bool:
        return True

    def _init_spd_factors(self, dim: int, scale):
        if isinstance(scale, torch.Tensor) and len(scale.shape):
            D, L = OKF.encode_SPD(scale.to(dtype=torch.double))
        else:
            D = (scale * (0.5 + torch.rand(dim, dtype=torch.double))).log()
            L = scale / 5 * torch.randn(dim * (dim - 1) // 2, dtype=torch.double)
        return D, L

    def reset_parameters(self) -> None:
        Q_D, Q_L = self._init_spd_factors(self.dim_x, self.Q0)
        R_D, R_L = self._init_spd_factors(self.dim_z, self.R0)
        if self.optimize:
            self.Q_D = nn.Parameter(Q_D, requires_grad=True)
            self.Q_L = nn.Parameter(Q_L, requires_grad=True)
            self.R_D = nn.Parameter(R_D, requires_grad=True)
            self.R_L = nn.Parameter(R_L, requires_grad=True)
        else:
            self.Q_D, self.Q_L = Q_D, Q_L
            self.R_D, self.R_L = R_D, R_L

    def process_covariance(self, x=None, z=None, r=None) -> torch.Tensor:
        return OKF.get_SPD(self.Q_D, self.Q_L)

    def observation_covariance(self, x=None, z=None, r=None) -> torch.Tensor:
        return OKF.get_SPD(self.R_D, self.R_L)

    def estimate_from_data(
        self, X, Z, true_fun=None, H=None, is_F_fun=True, is_H_fun=True
    ) -> None:
        X1 = torch.cat([torch.tensor(x[:-1], dtype=torch.double) for x in X], dim=0)
        X2 = torch.cat([torch.tensor(x[1:], dtype=torch.double) for x in X], dim=0)
        if is_F_fun:
            Fx1 = torch.stack([true_fun(x) for x in X1], dim=0)
        else:
            Fx1 = mp(true_fun, X1.T).T
        res = Fx1 - X2
        for i in range(res.shape[0]):
            utils.warpStateYawToPi(res[i])
        Q = torch.tensor(np.cov(res.T.detach().numpy()), dtype=torch.double)

        H_blocks = []
        Z_cat = np.concatenate(Z, axis=0)
        if is_H_fun:
            for x in X:
                H_blocks.extend(H(torch.tensor(x_t, dtype=torch.double)) for x_t in x)
        else:
            H_blocks = len(X) * [H]
        Hx = np.concatenate(
            [
                mp(h, torch.tensor(x, dtype=torch.double).T).T.detach().numpy()
                for x, h in zip(X, H_blocks)
            ],
            axis=0,
        )
        delta = Z_cat - Hx
        for i in range(delta.shape[0]):
            utils.warpResYawToPi(delta[i])
        R = torch.tensor(np.cov(delta.T), dtype=torch.double)

        Q_D, Q_L = OKF.encode_SPD(Q)
        R_D, R_L = OKF.encode_SPD(R)
        if self.optimize:
            with torch.no_grad():
                self.Q_D.copy_(Q_D)
                self.Q_L.copy_(Q_L)
                self.R_D.copy_(R_D)
                self.R_L.copy_(R_L)
        else:
            self.Q_D, self.Q_L = Q_D, Q_L
            self.R_D, self.R_L = R_D, R_L

    def export_state(self):
        return self.Q_D, self.Q_L, self.R_D, self.R_L

    def load_exported_state(self, state) -> None:
        Q_D, Q_L, R_D, R_L = state
        if self.optimize:
            with torch.no_grad():
                self.Q_D.copy_(Q_D)
                self.Q_L.copy_(Q_L)
                self.R_D.copy_(R_D)
                self.R_L.copy_(R_L)
        else:
            self.Q_D, self.Q_L = Q_D, Q_L
            self.R_D, self.R_L = R_D, R_L


class ProvidedRStaticQNoiseStrategy(NoiseStrategy):
    def __init__(self, dim_x: int, dim_z: int, optimize: bool, Q0=1):
        super().__init__(dim_x=dim_x, dim_z=dim_z, optimize=optimize)
        self.Q0 = Q0
        self.Q_D = None
        self.Q_L = None
        self.reset_parameters()

    @property
    def mode(self) -> NoiseStrategyMode:
        return NoiseStrategyMode.STATIC_Q_PROVIDED_R

    @property
    def has_trainable_q(self) -> bool:
        return True

    @property
    def requires_measurement_uncertainty(self) -> bool:
        return True

    def reset_parameters(self) -> None:
        if isinstance(self.Q0, torch.Tensor) and len(self.Q0.shape):
            Q_D, Q_L = OKF.encode_SPD(self.Q0.to(dtype=torch.double))
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

    def process_covariance(self, x=None, z=None, r=None) -> torch.Tensor:
        return OKF.get_SPD(self.Q_D, self.Q_L)

    def observation_covariance(self, x=None, z=None, r=None) -> torch.Tensor:
        if r is None:
            raise ValueError(
                "Measurement-provided observation noise requires per-step uncertainty 'r'."
            )
        r = _as_double_tensor(r)
        return torch.diag(r)

    def estimate_from_data(self, X, Z, true_fun=None, is_F_fun=True, **_) -> None:
        X1 = torch.cat([torch.tensor(x[:-1], dtype=torch.double) for x in X], dim=0)
        X2 = torch.cat([torch.tensor(x[1:], dtype=torch.double) for x in X], dim=0)
        if is_F_fun:
            Fx1 = torch.stack([true_fun(x) for x in X1], dim=0)
        else:
            Fx1 = mp(true_fun, X1.T).T
        res = Fx1 - X2
        for i in range(res.shape[0]):
            utils.warpStateYawToPi(res[i])
        Q = torch.tensor(np.cov(res.T.detach().numpy()), dtype=torch.double)
        Q_D, Q_L = OKF.encode_SPD(Q)
        if self.optimize:
            with torch.no_grad():
                self.Q_D.copy_(Q_D)
                self.Q_L.copy_(Q_L)
        else:
            self.Q_D, self.Q_L = Q_D, Q_L

    def export_state(self):
        return self.Q_D, self.Q_L

    def load_exported_state(self, state) -> None:
        Q_D, Q_L = state
        if self.optimize:
            with torch.no_grad():
                self.Q_D.copy_(Q_D)
                self.Q_L.copy_(Q_L)
        else:
            self.Q_D, self.Q_L = Q_D, Q_L


class NeuralNoiseStrategy(NoiseStrategy):
    FEATURE_INDICES = (6, 7, 8, 9)

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        optimize: bool,
        q_net: Optional[CovarianceMLP] = None,
        r_net: Optional[CovarianceMLP] = None,
        q_feature_indices: Optional[Sequence[int]] = None,
        r_feature_indices: Optional[Sequence[int]] = None,
        fallback_to_provided_r: bool = True,
        checkpoint_path: Optional[str] = None,
        hidden: int = 64,
        activation: str = "silu",
    ):
        super().__init__(dim_x=dim_x, dim_z=dim_z, optimize=optimize)
        self.q_feature_indices = tuple(q_feature_indices or self.FEATURE_INDICES)
        self.r_feature_indices = tuple(r_feature_indices or self.FEATURE_INDICES)
        self.fallback_to_provided_r = fallback_to_provided_r
        self.q_input_dim = len(self.q_feature_indices)
        self.r_input_dim = len(self.r_feature_indices)
        self.activation = activation
        self.q_net = q_net or CovarianceMLP(
            self.q_input_dim, dim_x, hidden=hidden, activation=activation
        )
        self.r_net = r_net or CovarianceMLP(
            self.r_input_dim, dim_z, hidden=hidden, activation=activation
        )
        self.double()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    @property
    def mode(self) -> NoiseStrategyMode:
        return NoiseStrategyMode.NEURAL

    @property
    def has_trainable_q(self) -> bool:
        return True

    @property
    def has_persistent_r(self) -> bool:
        return not self.fallback_to_provided_r

    @property
    def requires_measurement_uncertainty(self) -> bool:
        return self.fallback_to_provided_r

    def reset_parameters(self) -> None:
        for module in [self.q_net, self.r_net]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
                elif isinstance(layer, nn.LayerNorm):
                    layer.reset_parameters()

    def _extract_features(self, source, indices: Sequence[int]) -> torch.Tensor:
        if source is None:
            return torch.zeros(len(indices), dtype=torch.double)
        source = _as_double_tensor(source)
        feats = []
        for idx in indices:
            feats.append(
                source[idx]
                if idx < source.shape[0]
                else torch.tensor(0.0, dtype=torch.double)
            )
        return torch.stack(feats).to(dtype=torch.double)

    def _to_spd(self, L_or_flat: torch.Tensor, dim: int) -> torch.Tensor:
        if L_or_flat.dim() == 1:
            if L_or_flat.numel() != dim * (dim + 1) // 2:
                raise ValueError(
                    f"Expected flat Cholesky params of size {dim * (dim + 1) // 2}, "
                    f"got {L_or_flat.numel()}."
                )
            head = CholeskyHead(dim).to(dtype=L_or_flat.dtype, device=L_or_flat.device)
            L = head(L_or_flat)
        elif L_or_flat.dim() == 2 and L_or_flat.shape == (dim, dim):
            L = L_or_flat
        else:
            raise ValueError(
                f"Unsupported covariance representation with shape {tuple(L_or_flat.shape)}."
            )

        eps = 1e-3 * torch.eye(dim, dtype=L.dtype, device=L.device)
        return L @ L.T + eps

    def process_covariance(self, x=None, z=None, r=None) -> torch.Tensor:
        feats = self._extract_features(x, self.q_feature_indices)
        return self._to_spd(self.q_net(feats), self.dim_x)

    def observation_covariance(self, x=None, z=None, r=None) -> torch.Tensor:
        if self.fallback_to_provided_r and r is not None:
            return torch.diag(_as_double_tensor(r))
        feats_source = x if x is not None else z
        feats = self._extract_features(feats_source, self.r_feature_indices)
        return self._to_spd(self.r_net(feats), self.dim_z)

    def get_observation_covariance(self, to_numpy: bool = True):
        if self.fallback_to_provided_r:
            raise ValueError(
                "Neural strategy uses measurement-provided R at runtime; there is no single persistent R."
            )
        return super().get_observation_covariance(to_numpy=to_numpy)

    def save_checkpoint(self, path: str) -> None:
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_state_dict": self.q_net.state_dict(),
                "r_state_dict": self.r_net.state_dict(),
                "dim_x": self.dim_x,
                "dim_z": self.dim_z,
                "q_feature_indices": self.q_feature_indices,
                "r_feature_indices": self.r_feature_indices,
                "fallback_to_provided_r": self.fallback_to_provided_r,
                "q_hidden": self.q_net.hidden,
                "r_hidden": self.r_net.hidden,
                "q_activation": getattr(self.q_net, "activation", "silu"),
                "r_activation": getattr(self.r_net, "activation", "silu"),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        q_hidden = state.get("q_hidden", self.q_net.hidden)
        r_hidden = state.get("r_hidden", self.r_net.hidden)
        q_activation = state.get("q_activation", "silu")
        r_activation = state.get("r_activation", "silu")

        self.q_feature_indices = tuple(
            state.get("q_feature_indices", self.q_feature_indices)
        )
        self.r_feature_indices = tuple(
            state.get("r_feature_indices", self.r_feature_indices)
        )
        self.fallback_to_provided_r = state.get(
            "fallback_to_provided_r", self.fallback_to_provided_r
        )
        self.q_input_dim = len(self.q_feature_indices)
        self.r_input_dim = len(self.r_feature_indices)

        self.q_net = CovarianceMLP(
            self.q_input_dim,
            self.dim_x,
            hidden=q_hidden,
            activation=q_activation,
        ).double()
        self.r_net = CovarianceMLP(
            self.r_input_dim,
            self.dim_z,
            hidden=r_hidden,
            activation=r_activation,
        ).double()

        self.q_net.load_state_dict(state["q_state_dict"])
        self.r_net.load_state_dict(state["r_state_dict"])


class OKF(nn.Module):
    def __init__(
        self,
        motion_model: EKFMotionModel,
        model_name: str = "OKF",
        P0=1e3,
        Q0=1,
        R0=1,
        x0=None,
        optimize: bool = True,
        model_files_path: str = "models/",
        noise_strategy_mode: Optional[NoiseStrategyMode] = None,
        noise_strategy: Optional[NoiseStrategy] = None,
        neural_checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
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

        if noise_strategy_mode is None:
            noise_strategy_mode = NoiseStrategyMode.STATIC_LEARNED

        self.noise_strategy = noise_strategy or self._build_noise_strategy(
            noise_strategy_mode=noise_strategy_mode,
            Q0=Q0,
            R0=R0,
            neural_checkpoint_path=neural_checkpoint_path,
        )
        self.z2x = motion_model.initial_observation_to_state
        self.loss_fun = motion_model.loss_fun() or (
            lambda pred, x: ((pred - x) ** 2).sum()
        )

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

    def _build_noise_strategy(
        self, noise_strategy_mode, Q0, R0, neural_checkpoint_path=None
    ) -> NoiseStrategy:
        if noise_strategy_mode == NoiseStrategyMode.STATIC_LEARNED:
            return StaticNoiseStrategy(
                self.dim_x, self.dim_z, optimize=self.optimize, Q0=Q0, R0=R0
            )
        if noise_strategy_mode == NoiseStrategyMode.STATIC_Q_PROVIDED_R:
            return ProvidedRStaticQNoiseStrategy(
                self.dim_x, self.dim_z, optimize=self.optimize, Q0=Q0
            )
        if noise_strategy_mode == NoiseStrategyMode.NEURAL:
            return NeuralNoiseStrategy(
                self.dim_x,
                self.dim_z,
                optimize=self.optimize,
                checkpoint_path=neural_checkpoint_path,
            )
        raise ValueError(f"Unsupported noise strategy mode: {noise_strategy_mode}")

    def init_state(self):
        self.x = self.x0
        self.z = self.dim_z * [None]
        self.P = self.P0

    def reset_model(self):
        self.noise_strategy.reset_parameters()

    def save_model(self, fname=None, base_path=None, assert_suffices=True):
        fpath = self.get_model_path(fname, base_path, assert_suffices)
        if self.optimize:
            torch.save(self.state_dict(), fpath)
        else:
            torch.save(self.noise_strategy.export_state(), fpath)

    def load_model(self, fname=None, base_path=None, assert_suffices=True):
        fpath = self.get_model_path(fname, base_path, assert_suffices)
        if self.optimize:
            self.load_state_dict(torch.load(fpath, map_location="cpu"))
        else:
            self.noise_strategy.load_exported_state(
                torch.load(fpath, map_location="cpu")
            )

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
        Q = self.noise_strategy.process_covariance(x=self.x, z=self.z)
        self.x = self.true_fun(self.x)
        utils.warpStateYawToPi(self.x)
        self.P = mp(mp(F, self.P), F.T) + Q

    def update(self, z, r):
        self.z = _as_double_tensor(z)
        is_x_none = any(x is None for x in self.x)
        if is_x_none:
            H = (
                self.H(torch.tensor([0.0] * len(self.x), dtype=torch.double))
                if self.is_H_fun
                else self.H
            )
        else:
            H = self.H(self.x) if self.is_H_fun else self.H
        R = self.noise_strategy.observation_covariance(x=self.x, z=self.z, r=r)
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
        A = torch.linalg.cholesky(
            A + eps * torch.eye(n, dtype=A.dtype, device=A.device)
        )
        D = A.diag().log()
        ids = torch.tril_indices(n, n, -1)
        L = A[ids[0, :], ids[1, :]]
        return D, L

    def estimate_noise(self, X, Z):
        kwargs = dict(
            true_fun=self.true_fun,
            H=self.H,
            is_F_fun=self.is_F_fun,
            is_H_fun=self.is_H_fun,
        )
        self.noise_strategy.estimate_from_data(X, Z, **kwargs)

    def get_Q(self, to_numpy=True):
        return self.noise_strategy.get_process_covariance(to_numpy=to_numpy)

    def get_R(self, to_numpy=True):
        return self.noise_strategy.get_observation_covariance(to_numpy=to_numpy)

    def display_params(self, n_digits=0, fontsize=15, axsize=(4.5, 3.5)):
        mats = [("Q", self.get_Q())]
        try:
            mats.append(("R", self.get_R()))
        except Exception:
            pass
        axs = utils.Axes(1, len(mats), axsize=axsize)
        if not isinstance(axs, (list, tuple)):
            axs = [axs]
        for i, (name, A) in enumerate(mats):
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