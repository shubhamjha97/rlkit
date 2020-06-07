import numpy as np
import os, sys
import rlkit
from rlkit.environment import Environment
from rlkit.algorithms.policy_gradients import REINFORCE

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


value_estimator_specs = [
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
agent = REINFORCE(env_, network_specs, value_estimator_specs)
agent.train(episodes=1000, lr=0.001, gamma=1)