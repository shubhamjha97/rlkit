from rlkit.core.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, params, action_space):
        super(RandomAgent, self).__init__(params, action_space)

    def get_action(self, state):
        return self.action_space.sample()