import numpy as np
import os, sys
import RLkit
from RLkit.environment import Environment
from RLkit.algorithms import DQN

network_specs = [
  {
    "type": "dense",
    "size": 64,
    "activation":"sigmoid"
  },
  {
    "type": "dense",
    "size": 64,
    "activation":"relu"
  }
]
env_ = Environment(env_name="CartPole-v1", render = False)
agent = DQN(env_, network_specs, buffer_size = 100000, batch_size = 10, tau=0.001, update_target_every_n = 2000, eps=0.9, update_every_n = 200)
agent.train(env_, episodes=1000, lr=0.01, gamma=1)