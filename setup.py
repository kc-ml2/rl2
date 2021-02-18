from setuptools import setup

# TODO: add openai/baselines
# TODO: add torch (version1.4)
setup(
    name="rl2",
    install_requires=[
        'tqdm',
        'termcolor',
        'tensorboardX',
        'matplotlib',
        'pillow',
        'sklearn',
        'psutil',
        'mpi4py'
    ]
)
