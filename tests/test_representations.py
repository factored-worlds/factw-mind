import pytest

from factw.mind.representations import ObservedTransition, StateEstimate, estimate_state


class EvidenceEstimator:
    def initialize(self, observation):
        return StateEstimate(observation, ())

    def update(self, state, transition):
        evidence = (*state.memory, transition.outcome)
        return StateEstimate(transition.observation, evidence)


def test_estimation_consumes_only_completed_transitions_and_does_not_mutate_prior_state():
    estimator = EvidenceEstimator()
    initial = estimator.initialize("draft")
    observed = ObservedTransition("inspect", "passed", "unchanged draft", 1.0)
    following = estimator.update(initial, observed)
    assert initial == StateEstimate("draft", ())
    assert following == StateEstimate("unchanged draft", ("passed",))
    assert estimate_state(estimator, "draft", iter([observed])) == following
    assert estimate_state(estimator, "draft", ()) == initial


@pytest.mark.parametrize("elapsed", [True, -1, float("nan"), float("inf")])
def test_observed_transition_rejects_invalid_duration(elapsed):
    with pytest.raises(ValueError, match="Elapsed time"):
        ObservedTransition("read", "read", "draft", elapsed)
