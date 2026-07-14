from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from ivon import IVON as _UpstreamIVON


UpdateTransform = Literal["clip", "none", "muon_whiten"]


class IVON(_UpstreamIVON):
    """IVON with an explicit update-transform switch.

    The default ``update_transform="clip"`` follows upstream IVON's clipped
    preconditioned update and is the reproduction path for the existing runs.
    Other values are intentionally opt-in hooks for optimizer-internal
    experiments.
    """

    _MUON_EXCLUDED_NAME_FRAGMENTS = (
        "embed",
        "lm_head",
        "q_head",
    )

    def __init__(
        self,
        *args,
        update_transform: UpdateTransform = "clip",
        muon_whiten_eps: float = 1e-8,
        muon_ns_steps: int = 5,
        **kwargs,
    ) -> None:
        if update_transform not in ("clip", "none", "muon_whiten"):
            raise ValueError(f"Unsupported IVON update_transform: {update_transform}")
        if muon_whiten_eps <= 0:
            raise ValueError("muon_whiten_eps must be positive.")
        if muon_ns_steps <= 0:
            raise ValueError("muon_ns_steps must be positive.")
        self.update_transform = update_transform
        self.muon_whiten_eps = float(muon_whiten_eps)
        self.muon_ns_steps = int(muon_ns_steps)
        super().__init__(*args, **kwargs)

    def _transform_update(self, update: Tensor, clip_radius: float, group: dict | None = None) -> Tensor:
        if self.update_transform == "clip":
            return torch.clip(update, min=-clip_radius, max=clip_radius)
        if self.update_transform == "none":
            return update
        if group is None:
            raise ValueError("muon_whiten requires an optimizer parameter group.")
        return self._transform_muon_group_update(update, clip_radius, group)

    def _transform_muon_group_update(self, update: Tensor, clip_radius: float, group: dict) -> Tensor:
        pieces = []
        names = group.get("param_names")
        group_offset = 0
        param_index = 0
        for param in group["params"]:
            if param is None:
                continue
            param_update = update[group_offset : group_offset + param.numel()].view(param.shape)
            param_name = names[param_index] if names is not None and param_index < len(names) else None
            if self._should_muon_whiten(param, param_name):
                pieces.append(self._muon_orthogonalize(param_update).reshape(-1))
            else:
                pieces.append(torch.clip(param_update, min=-clip_radius, max=clip_radius).reshape(-1))
            group_offset += param.numel()
            param_index += 1
        assert group_offset == group["numel"]
        return torch.cat(pieces, 0)

    def _should_muon_whiten(self, param: Tensor, param_name: str | None) -> bool:
        if param.ndim != 2 or min(param.shape) <= 1:
            return False
        if param_name is None:
            return True
        normalized = param_name.removeprefix("_orig_mod.")
        return not any(fragment in normalized for fragment in self._MUON_EXCLUDED_NAME_FRAGMENTS)

    def _muon_orthogonalize(self, update: Tensor) -> Tensor:
        if not torch.isfinite(update).all():
            return update
        x = update.to(torch.float32)
        should_transpose = x.shape[0] > x.shape[1]
        if should_transpose:
            x = x.T

        x = x / x.norm().clamp_min(self.muon_whiten_eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(self.muon_ns_steps):
            xx_t = x @ x.T
            x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

        if should_transpose:
            x = x.T
        x = x * math.sqrt(max(1.0, update.shape[0] / update.shape[1]))
        return x.to(dtype=update.dtype)

    def _update(self) -> None:
        self.current_step += 1

        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat([p.flatten() for p in group["params"] if p is not None], 0)

            group["momentum"] = self._new_momentum(
                self.state["avg_grad"][pg_slice],
                group["momentum"],
                b1,
            )

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state["avg_gsq"],
                pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            debias = 1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0
            lr_scale = lr * (group["hess_init"] + group["weight_decay"]) if self.rescale_lr else lr
            raw_update = (group["momentum"] / debias + group["weight_decay"] * param_avg) / (
                group["hess"] + group["weight_decay"]
            )
            param_avg = param_avg - lr_scale * self._transform_update(raw_update, group["clip_radius"], group)

            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(p.shape)
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]
            offset += group["numel"]
        assert offset == self._numel
