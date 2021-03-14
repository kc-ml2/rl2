from setuptools import setup, find_packages

setup(
    name='rl2',
    version='0.0.0',
    url='https://github.com/kc-ml2/rl2',
    author_email='contact@kc-ml2.com',
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'termcolor',
        'tensorboardX',
        'matplotlib',
        'pillow',
        'sklearn',
        'psutil',
        'mpi4py',
        'easydict',
        'torch',
    ]
)


