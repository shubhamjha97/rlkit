from rlkit.core import BaseEnvironment
from vizdoom import *

class VizDoomEnvironment(BaseEnvironment):
    def __init__(self, params):
        super(VizDoomEnvironment, self).__init__()
        self.env_name = params["env_name"]

        pass

    def initialize_env(self):
        self.env = DoomGame()
        self.env.load_config("../config/basic.cfg")
        self.env.init()