import numpy as np
import os, sys
# sys.path.insert(0, 'C:/Users/Shubham/Documents/Shubham/Projects/rllib/rllib')
from RLkit.environment import Environment
from RLkit.algorithms import RandomAgent, REINFORCE, DQN

network_specs = [
  {
    "type": "dense",
    "size": 64,
    "activation":"relu"
  },
  {
    "type": "dense",
    "size": 64,
    "activation":"relu"
  }
]
env_ = Environment(env_name="CartPole-v1", render = False)
agent = DQN(env_, network_specs, buffer_size = 5000, batch_size = 128)
agent.train(env_, episodes=6000, lr=0.01, gamma=1)