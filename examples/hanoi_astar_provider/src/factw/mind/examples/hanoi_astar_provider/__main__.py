import json

from .demo import run_example

print(json.dumps(run_example(), indent=2))
