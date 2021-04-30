# RL2

---

**Reinforcement Learning Library** for deep RL algorithms

A opensource library built to provide a RL algorithm development framework. 

RL2 is built on the philosophy to unify different RL algorithm under a generalized core, to recycle codes across algorithms and make modifications to algorithms easy. 

### Installation

---

Clone RL2 repo and install using pip

```bash
git clone https://github.com/kc-ml2/rl2.git
cd rl2
pip install -e .

```

### Structure

---

![RL2](https://user-images.githubusercontent.com/5464491/116668295-34df1f80-a9d8-11eb-9b6b-86e5e10dc98e.png)

Simplified layout of components that consists the RL2 structure. 

- **Worker** : Governs the env-agent interaction loop
    - **Env** : Environment object
    - **Agent** : Algorithm specific agent that governs information from/to the environment
        - **Model** : Takes care of everything related to inferring and updating the neural network
        - **Memory** : Buffer to store all the information necessary for training.

Every agent in RL2 are designed to follow the above schematics. 

**BranchModel**

Branch model is a unique unit that carries two neural networks inside. 

The two neural networks are 'Encoder' and 'Head' and are placed in the following order. 

![single_branch](https://user-images.githubusercontent.com/5464491/116668425-60faa080-a9d8-11eb-9694-c90869273d67.png)

The input is encoded to intermediate representation through the Encoder and is processed through the Head to generate the output. 

Both Encoder and Head may be set with a custom neural network upon initialization. Otherwise, they will be set using default options which are

- Simple convolution with 3x3 kernel and maxpooling and a linear layer mapping to encoded_dim
- MLP mapping to encoded_dim

for the Encoder depending on the input type (image or vector).

The Head is a single Linear layer by default but can be easily changed by assigning a value >1 to the head_depth. 

**More complex architecture**

Each RL algorithm model may have more than one branch depending on its architecture. 

For example, PPO has a common encoder and separate heads for the policy and value functions. In this case, two branch models will be used with the same encoder networks. 

![double_branch](https://user-images.githubusercontent.com/5464491/116668524-78398e00-a9d8-11eb-9b86-bdcf3c9694c1.png)

When the two branches are run sequentially, the encoder will be run twice, which is computationally inefficient. 

In this case, the function that runs both branches at the same time can be decorated using

```python
from rl2.models.torch.base import TorchModel

# policy and value are instances of BranchModel

@TorchModel.sharedbranch
def forward(x):
	  pi = policy(x)
	  v = value(x)
	  return pi, v
```

Running the decorated function will trigger the encoders to save the output per input and use the previously inference intermediate representation for the same input. Therefore, running two branches sequentially for same input is now same as simply having the common encoder as the following. 

![mix_branch](https://user-images.githubusercontent.com/5464491/116668572-84bde680-a9d8-11eb-950f-42afa8768ba6.png)

In this manner, various types of RL network architectures can be built using multiple branch models. 

### Input arguments

---

The predefined algorithms in RL2 all inherit the core model and agent of TorchModel and Agent from rl2.models.torch.base and rl2.agents, respectively.

Each have common input arguments of the following. 

Model

- observation_shape : The observation shape of the environment to interact with
- action_shape : The action shape of the environment to interact with
- device : The device (cpu or cuda) in which the neural network will be placed

Agent

- model : The model object
- train_interval : Train interval for the model
- num_epochs : Number of epochs to train the model at every train_interval
- buffer_cls : Name of the buffer class from rl2.buffers
- buffer_kwargs : python dictionary for to pass to the buffer object on initialization

Please refer to the predefined algorithms **here** for examples of using these cores to create a new algorithm. 

### Basic usage

---

Simplest way of training on [OpenAI gym 'Cartpole'](https://gym.openai.com/envs/CartPole-v0/) using DQN

```python
import gym
from rl2.agents.dqn import DQNModel, DQNAgent
from rl2.workers import MaxStepWorker

# Create env
env = gym.make('Cartpole-v0')

# Create default DQN model
model = DQNModel(
	  env.observation_space.shape,
	  env.action_space.n,
	  default=True
)

# Create default DQN agent
agent = DQNAgent(model)

# Create and run max step worker
worker = MaxStepWorker(env, agent, max_steps=int(1e6), training=True)
worker.run() 
```

### Examples

---

Refer to [this link](https://github.com/kc-ml2/rl2/rl2/examples) for more example scripts for training the predefined algorithms.

**Predefined Algorithms**

- PPO
- DQN
    - DDQN
    - DRQN
- DDPG
- MADDPG

### Citation

---

```python
@MISC{rl2,
author =   {ML2},
title =    {RL2, Reinforcement Learning Library by ML2},
howpublished = {\url{http://github.com/kc-ml2/rl2}},
year = {2021}
}
```

### Updates

---
