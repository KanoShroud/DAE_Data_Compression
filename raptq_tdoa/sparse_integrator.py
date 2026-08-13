"""Numerically certified induced-likelihood integrator for sparse RAPTQ views.

This module is deliberately separate from :mod:`raptq_tdoa.core`.  The frozen
Stage 0 and Stage 0-I numerical-certificate entry points therefore retain their
original numerical behaviour.  The code below changes only the integration
coordinates and proposal used to evaluate the same sparse continuous
statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import betaincinv, gammaincinv, gammaln, logsumexp
from scipy.stats import qmc

from raptq_tdoa.core import BayesianRAPTQModel, ObservationContext


SparseRepresentation = Literal["sparse_phase", "sparse_amp_phase"]
SparseProposal = Literal["model", "defensive_mixture"]


@dataclass(frozen=True)
class SparseIntegrationResult:
    """Posterior and importance-sampling diagnostics for one configuration."""

    posterior: np.ndarray
    repeat_log_likelihoods: np.ndarray
    max_integrand_weight_fraction: float
    min_integrand_effective_sample_size: float


@dataclass(frozen=True)
class SparseQMCBatch:
    """One randomized-QMC batch in non-redundant sparse polar coordinates."""

    points: np.ndarray
    radius_squared: np.ndarray
    powers: np.ndarray
    phases: np.ndarray
    log_proposal_density: np.ndarray
    missing_phase_indices: np.ndarray
    proposal_component: np.ndarray


def dirichlet_stick_breaking(
    unit: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Map ``K-1`` independent uniforms to a ``K``-part Dirichlet vector."""

    values = np.asarray(unit, dtype=float)
    parameters = np.asarray(alpha, dtype=float)
    if parameters.ndim != 1 or parameters.size < 2:
        raise ValueError("alpha must contain at least two entries")
    if np.any(~np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("Dirichlet parameters must be positive and finite")
    if values.ndim != 2 or values.shape[1] != parameters.size - 1:
        raise ValueError("unit must have shape [n_points, len(alpha) - 1]")
    if np.any(~np.isfinite(values)) or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("unit coordinates must lie strictly inside (0, 1)")

    shares = np.empty((values.shape[0], parameters.size), dtype=float)
    remaining = np.ones(values.shape[0], dtype=float)
    tail = float(np.sum(parameters))
    for index in range(parameters.size - 1):
        tail -= float(parameters[index])
        fraction = betaincinv(parameters[index], tail, values[:, index])
        shares[:, index] = remaining * fraction
        remaining *= 1.0 - fraction
    shares[:, -1] = remaining
    if np.any(~np.isfinite(shares)) or np.any(shares <= 0.0):
        raise RuntimeError("stick-breaking produced invalid simplex coordinates")
    shares /= np.sum(shares, axis=1, keepdims=True)
    return shares


def log_dirichlet_density(shares: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Evaluate a Dirichlet density with respect to simplex Lebesgue measure."""

    values = np.asarray(shares, dtype=float)
    parameters = np.asarray(alpha, dtype=float)
    if values.ndim != 2 or parameters.shape != (values.shape[1],):
        raise ValueError("shares and alpha have incompatible shapes")
    if np.any(values <= 0.0) or np.any(~np.isfinite(values)):
        raise ValueError("shares must be positive and finite")
    if np.any(parameters <= 0.0) or np.any(~np.isfinite(parameters)):
        raise ValueError("alpha must be positive and finite")
    if not np.allclose(np.sum(values, axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("shares must sum to one")
    return (
        gammaln(float(np.sum(parameters)))
        - float(np.sum(gammaln(parameters)))
        + np.sum((parameters - 1.0) * np.log(values), axis=1)
    )


def _log_gamma_density(values: np.ndarray, shape: int, scale: float) -> np.ndarray:
    if shape < 1 or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Gamma shape and scale must be positive")
    return (
        (shape - 1.0) * np.log(values)
        - values / scale
        - gammaln(shape)
        - shape * math.log(scale)
    )


def _sample_wrapped_cauchy(
    unit: np.ndarray,
    centre: np.ndarray,
    rho: float,
) -> np.ndarray:
    if not 0.0 < rho < 1.0:
        raise ValueError("wrapped-Cauchy rho must lie in (0, 1)")
    scale = (1.0 - rho) / (1.0 + rho)
    offset = 2.0 * np.arctan(scale * np.tan(np.pi * (unit - 0.5)))
    return np.angle(np.exp(1j * (centre[None, :] + offset)))


def _log_wrapped_cauchy_density(
    phases: np.ndarray,
    centre: np.ndarray,
    rho: float,
) -> np.ndarray:
    delta = phases - centre[None, :]
    return np.sum(
        math.log1p(-(rho**2))
        - math.log(2.0 * math.pi)
        - np.log(1.0 + rho**2 - 2.0 * rho * np.cos(delta)),
        axis=1,
    )


class SparseRAPTQIntegrator:
    """Induced-likelihood integrator using non-redundant simplex coordinates."""

    def __init__(
        self,
        model: BayesianRAPTQModel,
        sparse_coordinate_count: int = 2,
        defensive_mixture_weight: float = 0.5,
        wrapped_cauchy_rho: float = 0.8,
    ) -> None:
        if not 1 <= sparse_coordinate_count < model.n_bins - 1:
            raise ValueError("invalid sparse coordinate count")
        if not math.isclose(defensive_mixture_weight, 0.5, abs_tol=1e-15):
            raise ValueError("the certified defensive mixture weight is fixed at 0.5")
        if not 0.0 < wrapped_cauchy_rho < 1.0:
            raise ValueError("wrapped-Cauchy rho must lie in (0, 1)")
        self.model = model
        self.sparse_coordinate_count = sparse_coordinate_count
        self.defensive_mixture_weight = defensive_mixture_weight
        self.wrapped_cauchy_rho = wrapped_cauchy_rho
        self.retained = np.arange(1, sparse_coordinate_count + 1, dtype=int)
        self.unretained = np.asarray(
            [index for index in range(model.n_bins) if index not in self.retained],
            dtype=int,
        )
        self.missing_phase_indices = np.asarray(
            [
                index
                for index in range(1, model.n_bins)
                if index not in self.retained
            ],
            dtype=int,
        )

    def _model_scale(self, context: ObservationContext) -> float:
        means, variances = self.model._means_and_variances(context)
        expected_energy = float(
            np.mean(np.sum(np.abs(means) ** 2, axis=2))
            + np.mean(np.sum(variances, axis=1))
        )
        return max(expected_energy / self.model.n_bins, 1e-3)

    def _coordinate_spec(
        self,
        context: ObservationContext,
        representation: SparseRepresentation,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        energy, observed_power, observed_phase = self.model.observed_coordinates(
            context.y_b_normalized
        )
        if representation == "sparse_phase":
            remaining_mass = 1.0
            observed_shares = observed_power
        elif representation == "sparse_amp_phase":
            remaining_mass = 1.0 - float(np.sum(observed_power[self.retained]))
            if remaining_mass <= np.finfo(float).eps:
                raise ValueError("unretained power mass is numerically zero")
            observed_shares = observed_power[self.unretained] / remaining_mass
        else:
            raise ValueError(f"unsupported sparse representation: {representation}")
        return (
            energy,
            observed_power,
            observed_phase,
            observed_shares,
            np.asarray([remaining_mass], dtype=float),
        )

    def _joint_log_density(
        self,
        radius_squared: np.ndarray,
        shares: np.ndarray,
        phases: np.ndarray,
        remaining_mass: float,
        model_scale: float,
        focused_scale: float,
        focused_alpha: np.ndarray,
        focused_phase: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dimension = shares.shape[1]
        model_alpha = np.ones(dimension, dtype=float)
        simplex_jacobian = (dimension - 1) * math.log(remaining_mass)
        phase_count = phases.shape[1]
        model = (
            _log_gamma_density(radius_squared, self.model.n_bins, model_scale)
            + log_dirichlet_density(shares, model_alpha)
            - simplex_jacobian
            - phase_count * math.log(2.0 * math.pi)
        )
        focused = (
            _log_gamma_density(radius_squared, self.model.n_bins, focused_scale)
            + log_dirichlet_density(shares, focused_alpha)
            - simplex_jacobian
            + _log_wrapped_cauchy_density(
                phases,
                focused_phase,
                self.wrapped_cauchy_rho,
            )
        )
        return model, focused

    def draw_batch(
        self,
        context: ObservationContext,
        representation: SparseRepresentation,
        power: int,
        seed: int,
        proposal: SparseProposal,
    ) -> SparseQMCBatch:
        if power < 4:
            raise ValueError("QMC power must be at least four")
        if proposal not in {"model", "defensive_mixture"}:
            raise ValueError("unsupported sparse proposal")
        (
            observed_energy,
            observed_power,
            observed_phase,
            observed_shares,
            remaining_array,
        ) = self._coordinate_spec(context, representation)
        remaining_mass = float(remaining_array[0])
        simplex_size = observed_shares.size
        phase_count = self.missing_phase_indices.size
        mixture_dimension = 1 if proposal == "defensive_mixture" else 0
        dimension = mixture_dimension + 1 + simplex_size - 1 + phase_count
        unit = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(power)
        unit = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        cursor = 0
        if proposal == "defensive_mixture":
            focused_component = unit[:, cursor] >= 0.5
            cursor += 1
        else:
            focused_component = np.zeros(unit.shape[0], dtype=bool)

        model_scale = self._model_scale(context)
        focused_scale = max(observed_energy / self.model.n_bins, 1e-3)
        selected_scale = np.where(focused_component, focused_scale, model_scale)
        radius_squared = selected_scale * gammaincinv(
            self.model.n_bins,
            unit[:, cursor],
        )
        cursor += 1

        model_alpha = np.ones(simplex_size, dtype=float)
        focused_alpha = 1.0 + simplex_size * observed_shares
        shares = np.empty((unit.shape[0], simplex_size), dtype=float)
        simplex_unit = unit[:, cursor : cursor + simplex_size - 1]
        cursor += simplex_size - 1
        for component in (False, True):
            mask = focused_component == component
            if np.any(mask):
                alpha = focused_alpha if component else model_alpha
                shares[mask] = dirichlet_stick_breaking(simplex_unit[mask], alpha)

        phase_unit = unit[:, cursor : cursor + phase_count]
        cursor += phase_count
        focused_phase = observed_phase[self.missing_phase_indices]
        phases_missing = 2.0 * np.pi * phase_unit - np.pi
        if np.any(focused_component):
            phases_missing[focused_component] = _sample_wrapped_cauchy(
                phase_unit[focused_component],
                focused_phase,
                self.wrapped_cauchy_rho,
            )
        if cursor != dimension:
            raise RuntimeError(
                f"sparse QMC coordinate mismatch: consumed {cursor}, expected {dimension}"
            )

        powers = np.empty((unit.shape[0], self.model.n_bins), dtype=float)
        phases = np.zeros((unit.shape[0], self.model.n_bins), dtype=float)
        phases[:, self.retained] = observed_phase[self.retained]
        phases[:, self.missing_phase_indices] = phases_missing
        if representation == "sparse_phase":
            powers[:] = shares
        else:
            powers[:, self.retained] = observed_power[self.retained]
            powers[:, self.unretained] = remaining_mass * shares
        points = np.sqrt(radius_squared[:, None] * powers) * np.exp(1j * phases)

        log_q_model, log_q_focused = self._joint_log_density(
            radius_squared,
            shares,
            phases_missing,
            remaining_mass,
            model_scale,
            focused_scale,
            focused_alpha,
            focused_phase,
        )
        if proposal == "model":
            log_proposal = log_q_model
        else:
            log_proposal = np.logaddexp(log_q_model, log_q_focused) - math.log(2.0)
        if np.any(~np.isfinite(log_proposal)) or np.any(~np.isfinite(points)):
            raise RuntimeError("sparse proposal produced non-finite values")
        return SparseQMCBatch(
            points=points,
            radius_squared=radius_squared,
            powers=powers,
            phases=phases,
            log_proposal_density=log_proposal,
            missing_phase_indices=self.missing_phase_indices.copy(),
            proposal_component=focused_component,
        )

    def integrate(
        self,
        context: ObservationContext,
        representation: SparseRepresentation,
        power: int,
        repeats: int,
        seed: int,
        proposal: SparseProposal,
    ) -> SparseIntegrationResult:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        repeat_log_likelihoods: list[np.ndarray] = []
        max_fraction = 0.0
        min_ess = math.inf
        for repeat in range(repeats):
            batch = self.draw_batch(
                context,
                representation,
                power,
                seed + 104729 * repeat,
                proposal,
            )
            log_integrand = (
                self.model.log_density_by_tau(batch.points, context)
                + (self.model.n_bins - 1.0)
                * np.log(batch.radius_squared)[:, None]
                - batch.log_proposal_density[:, None]
            )
            normalizer = logsumexp(log_integrand, axis=0)
            max_log_fraction = np.max(log_integrand, axis=0) - normalizer
            max_fraction = max(
                max_fraction,
                float(np.exp(np.max(max_log_fraction))),
            )
            ess = np.exp(
                2.0 * normalizer - logsumexp(2.0 * log_integrand, axis=0)
            )
            min_ess = min(min_ess, float(np.min(ess)))
            repeat_log_likelihoods.append(normalizer - math.log(batch.points.shape[0]))
        repeat_values = np.stack(repeat_log_likelihoods)
        log_likelihood = logsumexp(repeat_values, axis=0) - math.log(repeats)
        posterior = self.model._posterior_from_log_likelihood(log_likelihood)
        return SparseIntegrationResult(
            posterior=posterior,
            repeat_log_likelihoods=repeat_values,
            max_integrand_weight_fraction=max_fraction,
            min_integrand_effective_sample_size=min_ess,
        )

    def radial_measure_integral_check(
        self,
        context: ObservationContext,
        representation: SparseRepresentation,
        power: int,
        seed: int,
        proposal: SparseProposal,
    ) -> tuple[float, float]:
        """Compare an artificial radial integral against its analytic value."""

        batch = self.draw_batch(context, representation, power, seed, proposal)
        log_target = (
            -batch.radius_squared
            + (self.model.n_bins - 1.0) * np.log(batch.radius_squared)
            - batch.log_proposal_density
        )
        estimate = float(np.exp(logsumexp(log_target) - math.log(log_target.size)))
        phase_volume = (2.0 * math.pi) ** self.missing_phase_indices.size
        if representation == "sparse_phase":
            simplex_volume = 1.0 / math.factorial(self.model.n_bins - 1)
        else:
            _, observed_power, _ = self.model.observed_coordinates(
                context.y_b_normalized
            )
            remaining_mass = 1.0 - float(np.sum(observed_power[self.retained]))
            unretained_count = self.unretained.size
            simplex_volume = remaining_mass ** (unretained_count - 1) / math.factorial(
                unretained_count - 1
            )
        exact = (
            math.gamma(self.model.n_bins)
            * simplex_volume
            * phase_volume
        )
        return estimate, exact
