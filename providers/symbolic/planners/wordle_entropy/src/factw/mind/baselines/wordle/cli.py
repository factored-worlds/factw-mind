import argparse
import json
import secrets
from pathlib import Path

from factw.mind.representations import ObservedTransition

from .belief import ExactBeliefEstimator
from .policy import InformationPolicy, RandomPolicy
from .session import WordleSession


def main() -> None:
    parser = argparse.ArgumentParser(description="A supplied closed-loop symbolic Wordle example")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("information", "random-valid", "random-consistent"),
        default="information",
    )
    parser.add_argument("--seed", type=int, default=0, help="Public policy seed")
    parser.add_argument("--target-seed", type=int, help="Private environment seed; never exported")
    parser.add_argument(
        "--smsl", action="store_true", help="Monitor observed outcomes with optional State Language"
    )
    args = parser.parse_args()
    from factw.env.worlds.wordle import DEMO_TASK, Guess, WordleEnv

    env = WordleEnv(DEMO_TASK)
    env.reset(seed=secrets.randbits(64) if args.target_seed is None else args.target_seed)
    policy = (
        InformationPolicy(DEMO_TASK.allowed_guesses)
        if args.policy == "information"
        else RandomPolicy(
            DEMO_TASK.allowed_guesses, consistent=args.policy == "random-consistent", seed=args.seed
        )
    )
    session = WordleSession(ExactBeliefEstimator(DEMO_TASK.answers), policy, env.public_state())

    monitor = None
    if args.smsl:
        from factw.env.worlds.wordle.smsl import WordleMonitor

        monitor = WordleMonitor()

    def execute(action):
        result = env.step(Guess(**action))
        if monitor is not None:
            monitor.observe(result)
        return ObservedTransition(
            action, result.outcome, env.public_state(), result.elapsed_seconds
        )

    with args.output.open("x") as stream:
        while not (session.progress.terminated or session.progress.truncated):
            trace = session.step(execute)
            if monitor is not None:
                trace["monitor"] = monitor.facts()
            stream.write(json.dumps(trace, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "won": session.progress.won,
                "accepted_guesses": session.progress.accepted_guesses,
            }
        )
    )


if __name__ == "__main__":
    main()
