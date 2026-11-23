"""Use the canonical method's payload and semantic completion validators."""

from factw.mind.baselines.wordle.provider_contract import (
    validate_completion,
    validate_inputs,
    validate_outputs,
)

__all__ = ["validate_completion", "validate_inputs", "validate_outputs"]
