class BaseEnvironment:
    def __init__(self):
        self.to_render = False
        self.done = False
        self.reset()

    def close(self):
        pass

    def execute_action(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def setRender(self, to_render):
        self.to_render = to_render

    def get_action_space(self):
        pass
