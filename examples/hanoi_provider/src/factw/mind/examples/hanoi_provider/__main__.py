import json

from .demo import run_example


def main():
    print(json.dumps(run_example(), indent=2))


if __name__ == "__main__":
    main()
