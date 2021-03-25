from rl2.agents.dqn import DQNAgent, DQNModel
import time
import gym
import marlenv
from marlenv.wrappers import SingleAgent

env = gym.make('Snake-v1', num_snakes=1, num_fruits=100,
               width=30, height=30,
               vision_range=5)
env = SingleAgent(env)
observation_shape = env.observation_space.shape
action_shape = (env.action_space.n,)

model = DQNModel(observation_shape,
                 action_shape,
                 reorder=True,
                 discrete=True,
                 )

load_dir = '/home/eunki/rl2/rl2/examples/runs/DQN/SNAKE/VR/ED500K/batch1024/rdeps/20210321173432/ckpt/1000k/DQNModel.pt'
model.load(load_dir=load_dir)

agent = DQNAgent(model=model,
                 explore=False)

obs = env.reset()

for i in range(300):
    action = agent.act(obs)
    # print(action)
    obs, rew, done, info = env.step(action)
    time.sleep(0.25)
    env.render('ascii')
    # env.render('gif')
    if done:
        obs = env.reset()
