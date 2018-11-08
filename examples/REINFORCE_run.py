import numpy as np
import os, sys
sys.path.insert(0, 'C:/Users/Shubham/Documents/Shubham/Projects/rllib/rllib')
from environment import Environment
from algorithms import REINFORCE

network_specs = [
  {
    "type": "dense",
    "size": 64,
    "activation":"relu"
  },
  {
    "type": "dense",
    "size": 32,
    "activation":"relu"
  }
]
env_ = Environment(env_name="CartPole-v1", render = False)
agent = REINFORCE(env_, network_specs)
agent.train(episodes=6000, lr=0.01, gamma=1)