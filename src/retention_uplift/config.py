"""Shared project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

ACTIONS = ("control", "reminder", "voucher", "service_call")
ACTIVE_ACTIONS = ACTIONS[1:]


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    n_customers: int = 18_000
    n_waves: int = 24
    train_end_wave: int = 15
    validation_end_wave: int = 19
    action_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "control": 0.40,
            "reminder": 0.25,
            "voucher": 0.20,
            "service_call": 0.15,
        }
    )
    action_costs: dict[str, float] = field(
        default_factory=lambda: {
            "control": 0.0,
            "reminder": 0.50,
            "voucher": 10.0,
            "service_call": 6.0,
        }
    )
    capacity_shares: dict[str, float] = field(
        default_factory=lambda: {
            "reminder": 0.22,
            "voucher": 0.14,
            "service_call": 0.10,
        }
    )
    budget_per_customer: float = 0.35

    def validate(self) -> None:
        if self.n_customers < 1_000:
            raise ValueError("n_customers must be at least 1,000")
        if not 0 <= self.train_end_wave < self.validation_end_wave < self.n_waves - 1:
            raise ValueError("wave boundaries must leave non-empty train, validation, and test")
        if set(self.action_probabilities) != set(ACTIONS):
            raise ValueError("action probabilities must cover every action")
        if any(not isfinite(value) for value in self.action_probabilities.values()):
            raise ValueError("action probabilities must be finite")
        if abs(sum(self.action_probabilities.values()) - 1.0) > 1e-9:
            raise ValueError("action probabilities must sum to one")
        if any(value <= 0 for value in self.action_probabilities.values()):
            raise ValueError("every randomized arm must have positive probability")
        if set(self.action_costs) != set(ACTIONS):
            raise ValueError("action costs must cover every action")
        if any(not isfinite(value) or value < 0 for value in self.action_costs.values()):
            raise ValueError("action costs must be finite and non-negative")
        if self.action_costs["control"] != 0:
            raise ValueError("control must have zero treatment cost")
        if set(self.capacity_shares) != set(ACTIVE_ACTIONS):
            raise ValueError("capacity shares must cover every active action")
        if any(
            not isfinite(value) or not 0 <= value <= 1
            for value in self.capacity_shares.values()
        ):
            raise ValueError("capacity shares must be finite and between zero and one")
        if not isfinite(self.budget_per_customer) or self.budget_per_customer < 0:
            raise ValueError("budget_per_customer must be finite and non-negative")
