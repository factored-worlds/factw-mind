from dataclasses import dataclass

from factw.mind.representations import ObservedTransition

from .rules import feedback, vocabulary, word


class ContradictionError(ValueError):
    """No answer in the declared lexicon explains the supplied feedback."""


@dataclass(frozen=True)
class Evidence:
    guess: str
    feedback: tuple[int, ...]


@dataclass(frozen=True)
class Belief:
    candidates: tuple[str, ...]
    evidence: tuple[Evidence, ...]


def public_history(public: dict) -> tuple[Evidence, ...]:
    fields = {
        "guesses",
        "feedbacks",
        "attempts",
        "max_guesses",
        "max_attempts",
        "won",
        "terminated",
        "truncated",
    }
    if type(public) is not dict or set(public) != fields:
        raise ValueError("Expected only the declared public Wordle fields")
    if type(public["guesses"]) is not list or type(public["feedbacks"]) is not list:
        raise ValueError("Public Wordle history must contain arrays")
    if len(public["guesses"]) != len(public["feedbacks"]):
        raise ValueError("Guesses and feedback histories differ in length")
    for name in ("attempts", "max_guesses", "max_attempts"):
        if type(public[name]) is not int or public[name] < (0 if name == "attempts" else 1):
            raise ValueError("Invalid public Wordle counter")
    if not len(public["guesses"]) <= public["attempts"] <= public["max_attempts"]:
        raise ValueError("Public attempt count disagrees with history")
    if len(public["guesses"]) > public["max_guesses"]:
        raise ValueError("Public history exceeds accepted guess limit")
    for name in ("won", "terminated", "truncated"):
        if type(public[name]) is not bool:
            raise ValueError("Wordle progress fields must be Boolean")
    history = []
    for guess, codes in zip(public["guesses"], public["feedbacks"], strict=True):
        word(guess)
        if (
            type(codes) is not list
            or len(codes) != 5
            or any(type(c) is not int or c not in (0, 1, 2) for c in codes)
        ):
            raise ValueError("Feedback must contain five integer codes in 0..2")
        if history and history[-1].feedback == (2,) * 5:
            raise ValueError("Public history continues after a win")
        history.append(Evidence(guess, tuple(codes)))
    won = bool(history and history[-1].feedback == (2,) * 5)
    terminated = won or len(history) == public["max_guesses"]
    truncated = public["attempts"] == public["max_attempts"] and not terminated
    if (public["won"], public["terminated"], public["truncated"]) != (won, terminated, truncated):
        raise ValueError("Public progress disagrees with feedback and limits")
    return tuple(history)


class ExactBeliefEstimator:
    def __init__(self, answers: tuple[str, ...]):
        self.answers = vocabulary(answers)

    def initialize(self, observation: dict) -> Belief:
        return self._filter(self.answers, public_history(observation))

    def update(self, state: Belief, transition: ObservedTransition) -> Belief:
        history = public_history(transition.observation)
        if type(transition.action) is not dict or set(transition.action) != {"guess"}:
            raise ValueError("Expected a guess command")
        guess = word(transition.action["guess"])
        if transition.outcome == "illegal":
            if history != state.evidence:
                raise ValueError("Rejected guess changed public evidence")
            return state
        if transition.outcome not in {"guessed", "won", "lost"}:
            raise ValueError("Unknown Wordle outcome")
        if history[:-1] != state.evidence or not history or history[-1].guess != guess:
            raise ValueError("Completed transition does not extend the current public history")
        public = transition.observation
        expected = "won" if public["won"] else "lost" if public["terminated"] else "guessed"
        if transition.outcome != expected:
            raise ValueError("Observed outcome disagrees with public feedback")
        result = self._filter(state.candidates, history[-1:])
        return Belief(result.candidates, history)

    @staticmethod
    def _filter(candidates: tuple[str, ...], history: tuple[Evidence, ...]) -> Belief:
        consistent = tuple(
            candidate
            for candidate in candidates
            if all(feedback(turn.guess, candidate) == turn.feedback for turn in history)
        )
        if not consistent:
            raise ContradictionError("No declared answer is consistent with public feedback")
        return Belief(consistent, history)
