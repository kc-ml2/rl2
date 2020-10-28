# ML2-RL2 (Reinforcement Learning Library)

### Running
Please run the following to install dependencies
```
python setup.py install
```
Then, install openai baselines from 
https://github.com/openai/baselines
[[commit that was used](https://github.com/openai/baselines/tree/ea25b9e8b234e6ee1bca43083f8f3cf974143998)]


---
Sample command for running a model on atari breakout
```
python main.py --mode=ppo --env=atari --env_id=Breakout --tag=test_run
```
