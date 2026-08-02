"""Shared project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

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
        if not self.train_end_wave < self.validation_end_wave < self.n_waves - 1:
            raise ValueError("wave boundaries must leave non-empty train, validation, and test")
        if set(self.action_probabilities) != set(ACTIONS):
            raise ValueError("action probabilities must cover every action")
        if abs(sum(self.action_probabilities.values()) - 1.0) > 1e-9:
            raise ValueError("action probabilities must sum to one")
        if any(value <= 0 for value in self.action_probabilities.values()):
            raise ValueError("every randomized arm must have positive probability")
