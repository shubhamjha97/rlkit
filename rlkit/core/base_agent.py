class BaseAgent:
    def __init__(self, params, action_space):
        self.params = params
        self.action_space = action_space

    def train(self):
        pass

    def get_action(self, state):
        pass