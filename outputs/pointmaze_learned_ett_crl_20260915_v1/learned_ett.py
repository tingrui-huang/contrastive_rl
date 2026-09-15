"""Learned stochastic F4 transition backend for the bounded PointMaze pilot.

The module has no environment import.  It predicts live XY, enforces only a
static endpoint-occupancy check, shifts visible F4 history exactly, and carries
an internal sampled failure state persistently.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WALLS = np.array([
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
], dtype=np.int64)


def legal_numpy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy)
    bounded = (
        (xy[..., 0] >= 0.0) & (xy[..., 0] < 9.0)
        & (xy[..., 1] >= 0.0) & (xy[..., 1] < 5.0)
    )
    cell = np.floor(xy).astype(np.int64)
    safe = np.clip(cell, [0, 0], [8, 4])
    return bounded & (WALLS[safe[..., 0], safe[..., 1]] == 0)


class LearnedETT(nn.Module):
    """Conditional implicit movement generator plus probabilistic onset head."""

    feature_dim = 18

    def __init__(self, state_mean, state_std, max_displacement=1.25, noise_dim=4):
        super().__init__()
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        self.max_displacement = float(max_displacement)
        self.noise_dim = int(noise_dim)
        self.move_context = nn.Sequential(
            nn.Linear(self.feature_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.generator = nn.Sequential(
            nn.Linear(128 + self.noise_dim, 128), nn.SiLU(), nn.Linear(128, 2),
        )
        self.onset_context = nn.Sequential(
            nn.Linear(self.feature_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.onset_head = nn.Sequential(
            nn.Linear(64 + 2, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    def features(self, state, xb, xq):
        state = (state - self.state_mean) / self.state_std
        delta = xq - xb
        return torch.cat([state, xb, xq, delta, xq * xb, delta.square()], dim=-1)

    @staticmethod
    def legal_torch(xy):
        bounded = (
            (xy[..., 0] >= 0.0) & (xy[..., 0] < 9.0)
            & (xy[..., 1] >= 0.0) & (xy[..., 1] < 5.0)
        )
        cell = torch.floor(xy).to(torch.long)
        cx = cell[..., 0].clamp(0, 8)
        cy = cell[..., 1].clamp(0, 4)
        walls = torch.as_tensor(WALLS, device=xy.device)
        return bounded & (walls[cx, cy] == 0)

    def movement_samples(self, state, xb, xq, count, generator=None):
        """Return emitted displacement samples [B,K,2] and raw legality mask."""
        feat = self.features(state, xb, xq)
        context = self.move_context(feat)
        shape = (len(state), int(count), self.noise_dim)
        z = torch.randn(shape, dtype=state.dtype, device=state.device, generator=generator)
        tiled = context[:, None, :].expand(-1, int(count), -1)
        raw = self.max_displacement * torch.tanh(
            self.generator(torch.cat([tiled, z], dim=-1))
        )
        proposal = state[:, None, :2] + raw
        legal = self.legal_torch(proposal)
        emitted = torch.where(legal[..., None], raw, torch.zeros_like(raw))
        return emitted, legal

    def onset_logits(self, state, xb, xq, displacement):
        feat = self.features(state, xb, xq)
        context = self.onset_context(feat)
        return self.onset_head(torch.cat([context, displacement], dim=-1)).squeeze(-1)

    def onset_probability(self, state, xb, xq, displacement):
        return torch.sigmoid(self.onset_logits(state, xb, xq, displacement))

    def sample_live(self, state, xb, xq, generator=None):
        displacement, legal = self.movement_samples(state, xb, xq, 1, generator)
        displacement = displacement[:, 0]
        probability = self.onset_probability(state, xb, xq, displacement)
        uniform = torch.rand(probability.shape, device=probability.device, generator=generator)
        event = uniform < probability
        next_xy = state[:, :2] + displacement
        successor = torch.cat([next_xy, state[:, :6]], dim=-1)
        return successor, event, probability, legal[:, 0]


def energy_score(samples, target):
    """Full emitted-XY Energy Score U-statistic, returned per row."""
    first = torch.linalg.vector_norm(samples - target[:, None, :], dim=-1).mean(dim=1)
    count = samples.shape[1]
    pair = torch.cdist(samples, samples)
    second = pair.sum(dim=(1, 2)) / (2.0 * count * (count - 1))
    return first - second


def state_dict_vector(state_dict):
    pieces = [value.detach().cpu().reshape(-1).to(torch.float64)
              for name, value in state_dict.items() if not name.endswith(("state_mean", "state_std"))]
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float64)


def _to_device(data, names, device):
    return {name: torch.as_tensor(data[name], dtype=torch.float32, device=device) for name in names}


@torch.no_grad()
def evaluate_model(model, data, count, seed, device, batch_size=512, return_samples=False):
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    scores, probabilities, predicted, samples_all, legal_all = [], [], [], [], []
    for start in range(0, len(data["s"]), batch_size):
        stop = min(start + batch_size, len(data["s"]))
        batch = _to_device({k: v[start:stop] for k, v in data.items()},
                           ("s", "xb", "xq", "y"), device)
        samples, legal = model.movement_samples(
            batch["s"], batch["xb"], batch["xq"], count, generator
        )
        target_delta = batch["y"][:, :2] - batch["s"][:, :2]
        scores.append(energy_score(samples, target_delta).cpu().numpy())
        actual_logits = model.onset_logits(
            batch["s"], batch["xb"], batch["xq"], target_delta
        )
        probabilities.append(torch.sigmoid(actual_logits).cpu().numpy())
        predicted.append(samples.mean(dim=1).cpu().numpy())
        legal_all.append(legal.cpu().numpy())
        if return_samples:
            samples_all.append(samples.cpu().numpy())
    result = {
        "score": np.concatenate(scores),
        "onset_probability_actual_successor": np.concatenate(probabilities),
        "predicted_displacement": np.concatenate(predicted),
        "proposal_legal": np.concatenate(legal_all),
    }
    if return_samples:
        result["samples"] = np.concatenate(samples_all)
    return result


def train_model(config, train, validation, initial_path: Path, selected_path: Path,
                last_path: Path, device, progress=None):
    """Run the one fixed supervised schedule and return its auditable summary."""
    torch.manual_seed(int(config["seed"]))
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    model = LearnedETT(
        initial["state_mean"], initial["state_std"],
        max_displacement=config["model"]["max_displacement"],
        noise_dim=config["model"]["noise_dim"],
    ).to(device)
    model.load_state_dict(initial["state_dict"])
    initial_vector = state_dict_vector(model.state_dict())
    initial_move = torch.cat([p.detach().cpu().reshape(-1).double()
                              for p in list(model.move_context.parameters()) + list(model.generator.parameters())])
    initial_onset = torch.cat([p.detach().cpu().reshape(-1).double()
                               for p in list(model.onset_context.parameters()) + list(model.onset_head.parameters())])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    diag = np.max(np.abs(train["xq"] - train["xb"]), axis=1) <= config["diagonal_tolerance"]
    diag_ids, off_ids = np.flatnonzero(diag), np.flatnonzero(~diag)
    positive_ids = np.flatnonzero(train["onset"])
    negative_ids = np.flatnonzero(~train["onset"])
    if min(len(diag_ids), len(off_ids), len(positive_ids), len(negative_ids)) == 0:
        raise RuntimeError("a required supervised stratum is empty")
    prevalence = float(train["onset"].mean())
    rng = np.random.default_rng(int(config["seed"]) + 1)
    movement_count = int(config["training"]["movement_samples"])
    batch_size = int(config["training"]["batch_per_movement_term"])
    onset_size = int(config["training"]["onset_batch_per_class"])
    onset_weight = float(config["training"]["onset_loss_weight"])
    updates = int(config["training"]["updates"])
    validate_every = int(config["training"]["validation_every"])
    history, best = [], None
    t0 = __import__("time").time()

    def batch(indices):
        return _to_device({k: train[k][indices] for k in ("s", "xb", "xq", "y")},
                          ("s", "xb", "xq", "y"), device)

    for step in range(1, updates + 1):
        model.train()
        d = batch(rng.choice(diag_ids, batch_size, replace=True))
        o = batch(rng.choice(off_ids, batch_size, replace=True))
        pids = rng.choice(positive_ids, onset_size, replace=True)
        nids = rng.choice(negative_ids, onset_size, replace=True)
        p, n = batch(pids), batch(nids)
        ds, _ = model.movement_samples(d["s"], d["xb"], d["xq"], movement_count)
        os, _ = model.movement_samples(o["s"], o["xb"], o["xq"], movement_count)
        dtarget = d["y"][:, :2] - d["s"][:, :2]
        otarget = o["y"][:, :2] - o["s"][:, :2]
        diag_loss = energy_score(ds, dtarget).mean()
        off_loss = energy_score(os, otarget).mean()
        pdisp = (p["y"][:, :2] - p["s"][:, :2]).detach()
        ndisp = (n["y"][:, :2] - n["s"][:, :2]).detach()
        positive_bce = F.binary_cross_entropy_with_logits(
            model.onset_logits(p["s"], p["xb"], p["xq"], pdisp),
            torch.ones(onset_size, device=device),
        )
        negative_bce = F.binary_cross_entropy_with_logits(
            model.onset_logits(n["s"], n["xb"], n["xq"], ndisp),
            torch.zeros(onset_size, device=device),
        )
        onset_loss = prevalence * positive_bce + (1.0 - prevalence) * negative_bce
        loss = diag_loss + off_loss + onset_weight * onset_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if not math.isfinite(float(loss)):
            raise FloatingPointError(f"non-finite loss at update {step}")

        if step == 1 or step % validate_every == 0 or step == updates:
            ev = evaluate_model(
                model, validation, count=32, seed=int(config["seed"]) + 1000,
                device=device, return_samples=False,
            )
            vdiag = np.max(np.abs(validation["xq"] - validation["xb"]), axis=1) <= config["diagonal_tolerance"]
            target = validation["onset"].astype(np.float64)
            prob = np.clip(ev["onset_probability_actual_successor"], 1e-8, 1 - 1e-8)
            bce = float(-(target * np.log(prob) + (1 - target) * np.log1p(-prob)).mean())
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "train_diagonal_es": float(diag_loss.detach().cpu()),
                "train_off_diagonal_es": float(off_loss.detach().cpu()),
                "train_onset_bce": float(onset_loss.detach().cpu()),
                "gradient_l2_before_clip": float(grad_norm.detach().cpu()),
                "validation_diagonal_es": float(ev["score"][vdiag].mean()),
                "validation_off_diagonal_es": float(ev["score"][~vdiag].mean()),
                "validation_onset_bce": bce,
            }
            record["selection_score"] = (
                record["validation_diagonal_es"] + record["validation_off_diagonal_es"]
                + onset_weight * bce
            )
            history.append(record)
            if best is None or record["selection_score"] < best["record"]["selection_score"]:
                best = {"record": copy.deepcopy(record),
                        "state_dict": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            if progress:
                progress(record)

    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "state_mean": initial["state_mean"], "state_std": initial["state_std"],
        "step": updates, "config": config,
    }, last_path)
    assert best is not None
    torch.save({
        "state_dict": best["state_dict"],
        "state_mean": initial["state_mean"], "state_std": initial["state_std"],
        "step": best["record"]["step"], "config": config,
    }, selected_path)

    model.load_state_dict(best["state_dict"])
    final_vector = state_dict_vector(model.state_dict())
    final_move = torch.cat([p.detach().cpu().reshape(-1).double()
                            for p in list(model.move_context.parameters()) + list(model.generator.parameters())])
    final_onset = torch.cat([p.detach().cpu().reshape(-1).double()
                             for p in list(model.onset_context.parameters()) + list(model.onset_head.parameters())])
    return {
        "updates": updates,
        "selected_step": best["record"]["step"],
        "selected_validation": best["record"],
        "history": history,
        "parameter_delta_l2": float(torch.linalg.vector_norm(final_vector - initial_vector)),
        "movement_parameter_delta_l2": float(torch.linalg.vector_norm(final_move - initial_move)),
        "onset_parameter_delta_l2": float(torch.linalg.vector_norm(final_onset - initial_onset)),
        "parameters_finite": bool(torch.isfinite(final_vector).all()),
        "wall_seconds": __import__("time").time() - t0,
        "device": str(device),
    }


def load_checkpoint(path: Path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = LearnedETT(
        payload["state_mean"], payload["state_std"],
        max_displacement=config["model"]["max_displacement"],
        noise_dim=config["model"]["noise_dim"],
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
