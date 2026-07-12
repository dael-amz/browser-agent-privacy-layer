"""Inference-provider presets shared by the launcher and proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    base_url: str
    model: str
    key_names: tuple[str, ...]
    models: tuple[str, ...] = ()

    def allowed_models(self) -> tuple[str, ...]:
        return self.models if self.model in self.models else (self.model, *self.models)


PROVIDERS: Final[dict[str, ProviderSpec]] = {
    "overshoot": ProviderSpec(
        base_url="https://api.overshoot.ai/v1",
        model="Hcompany/Holo3-35B-A3B",
        key_names=("OVERSHOOT_API_KEY", "API_KEY"),
        models=(
            "Hcompany/Holo3-35B-A3B",
            "Hcompany/Holo-3.1-35B-A3B-FP8",
            "Holo3-122B-A10B",
        ),
    ),
    "hcompany": ProviderSpec(
        base_url="https://api.hcompany.ai/v1",
        model="holo3-1-35b-a3b",
        key_names=("HAI_API_KEY",),
        models=("holo3-1-35b-a3b",),
    ),
}
