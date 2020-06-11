class BaseEnvironment:
    def __init__(self):
        self.to_render = False
        self.reset()

    def execute_action(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def setRender(self, to_render):
        self.to_render = to_render
