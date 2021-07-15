import runpy
from glob import glob

examples = glob('../rl2/examples/*.py')
for example in examples:
    runpy.run_path(example)