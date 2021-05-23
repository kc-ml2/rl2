from rl2.agents.dqn import DQNAgent, DQNModel
import time
import gym
import marlenv
from marlenv.wrappers import SingleAgent
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

env = gym.make('Snake-v1',
               num_snakes=1,
               num_fruits=4,
               width=10,
               height=10,
               frame_stack=2,
               vision_range=5,
               )
env = SingleAgent(env)
observation_shape = env.observation_space.shape
action_shape = (env.action_space.n,)

model = DQNModel(observation_shape,
                 action_shape,
                 reorder=True,
                 discrete=True,
                 )

load_dir = '/home/eunki/rl2/rl2/examples/runs/DDQN/SNAKE/FS2/MS20NF4/20210325163351/ckpt/3000k/DQNModel.pt'
model.load(load_dir=load_dir)

agent = DQNAgent(model=model,
                 explore=False)

obs = env.reset()

for i in range(500):
    action = agent.act(obs)
    # print(action)
    obs, rew, done, info = env.step(action)
    time.sleep(0.25)
    env.render('ascii')
    # env.render('gif')
    if done:
        obs = env.reset()
