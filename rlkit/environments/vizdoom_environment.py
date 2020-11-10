import random
import time

from rlkit.core import BaseEnvironment, BaseActionSpace
from vizdoom import *

class VizDoomEnvironment(BaseEnvironment):

    class VizDoomActionSpace(BaseActionSpace):
        def __init__(self):
            self.actions = [
                # http://www.cs.put.poznan.pl/visualdoomai/tutorial.html
                [0, 0, 1], # shoot
                [1, 0, 0], # left
                [0, 1, 0], # right
            ]
            super(VizDoomEnvironment.VizDoomActionSpace, self).__init__()

        def sample(self):
            return random.sample(self.actions, 1)[0]

    def __init__(self, params):
        self.action_space = self.VizDoomActionSpace()
        self.initialize_env()
        super(VizDoomEnvironment, self).__init__()

    def initialize_env(self):
        self.env = DoomGame()
        self.env.load_config("./basic.cfg") # TODO: load via params
        self.env.init()

    def get_action_space(self):
        return self.action_space

    def reset(self, reset_values=True):
        if reset_values:
            self.reset_values()
        self.reset_env()

    def reset_values(self):
        self.state = None
        self.reward = None
        self.done = False
        self.info = None

    def reset_env(self):
        self.env.new_episode()

    def step(self, action):
        self.reward = self.env.make_action(action)

        # TODO: see if need to get image buffer
        # TODO: see if this happens before/after reward
        self.state = self.env.get_state()

        self.done = self.env.is_episode_finished()
        if not self.done:
            self.info = self.state.game_variables
        else:
            self.info = None

        print(action, self.done, self.env.get_total_reward(), self.info)
        time.sleep(0.02) # TODO: remove
        return (self.state, self.reward, self.done, self.info, )
