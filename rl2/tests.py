import gym
from rl2.algos.ppo import PPOAgent
from rl2.distributions import CategoricalHead, ScalarHead
from rl2.managers import PGManager
from rl2.models.torch.actor_critic import ActorCritic
from rl2.networks.torch.networks import DeepMindEnc
from rl2.workers.base import EpisodicWorker
#
# class QValueModel(TorchModel):
#     def __init__(self, input_shape,  device):
#         self.encoder = DeepMindEnc(input_shape).to(device)
#         self.q_head = ScalarHead(self.encoder.out_shape, action_space.n).to(device)
#
#     def update_target(self):
#         self.copy_param()
#
#
# env = gym.make('CartPole-v0')
#
# model = Model()
# # model.optimizer1 = torch.optim.Adam([encoder.parameters(), actor.parameters()])
# # model.optimizer2 = torch.optim.Adam([encoder.parameters(), critic.parameters()])
# agent = Agent(model)
# trainer = SingleAgentTrainer(env, agent, max_steps=1e7)
# trainer.run()
#
# ############################### train custom client w/ buffer
#
# encoder = nn.Linear()
# actor = nn.Linear()
# critic = nn.Linear()
#
# env = gym.make('CartPole-v0')
#
# model = Model()
# agent = Agent(model)
# agent.buffer = Buffer()
# agent.manager = BufferManager()
#
# trainer = SingleAgentTrainer(env, agent, max_steps=1e7)
# trainer.run()

############################### train client w/ buffer
from rl2.workers.base import MaxStepWorker

env = gym.make('Snake-v1')

input_shape = env.observation_space.shape
if len(input_shape) > 1:
    input_shape = (input_shape[-1], *input_shape[:-1])
encoder = DeepMindEnc(input_shape)
actor = CategoricalHead(encoder.out_shape, env.action_space.n)
critic = ScalarHead(encoder.out_shape, 1)
model = ActorCritic(encoder=encoder, actor=actor, critic=critic)
manager = PGManager()
agent = PPOAgent(model, manager)

worker = MaxStepWorker(env, agent, training=True, max_steps=1e7)
worker.run()

worker = EpisodicWorker(env, agent, training=False, num_episodes=1, render=True)
worker.run()

############################### test client
#
# env = gym.make('CartPole-v0')
# agent = PPOAgent()
#
# worker = EpisodicWorker(env, agent, num_episodes=1, render=True)
# # render can be set later
# worker.run(render=True)

############################## SNL

