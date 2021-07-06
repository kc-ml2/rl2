from setuptools import setup, find_packages


setup(
    name='rl2',
    version='1.0.0',
    url='https://github.com/kc-ml2/rl2',
    long_description=open('README.md').read(),
    author='Daniel Nam, Won Seok Jung',
    author_email='contact@kc-ml2.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch==1.4',
        'numpy',
        'pyyaml',
        'gin-config',
        'click',
        'tqdm',
        'mpi4py',
        'absl-py',
        'tensorboard',
        'marlenv @ git+https://github.com/kc-ml2/marlenv.git',
        'matplotlib', 'moviepy', 'pillow<=7.0.0',
    ],
    # dependency_links=[
    # ],
    # extras_require={
    #     'visual': ['matplotlib', 'moviepy', 'pillow<=7.0.0', ]
    # },
    entry_points={
        'console_scripts': ['rl2=rl2.cli:main']
    }
)
