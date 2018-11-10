import numpy as np
import os, sys
import RLkit
from RLkit.environment import Environment
from RLkit.algorithms.policy_gradients import ActorCritic

actor_specs = [
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

critic_specs = [
  {
    "type": "dense",
    "size": 20,
    "activation":"sigmoid"
  }
]

env_ = Environment(env_name="CartPole-v1", render = False)
agent = ActorCritic(env_, actor_specs, critic_specs)
agent.train(episodes=1000, actor_lr=0.001, critic_lr=0.1, gamma=1)