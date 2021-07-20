import runpy
from glob import glob

examples = glob('../rl2/examples/*.py')
for example in examples:
    # pass config that runs training only few steps/episodes
    runpy.run_path(example)