class BaseEnvironment:
    def __init__(self, params, metrics):
        if not hasattr(self, "params"):
            self.params = params
        if not hasattr(self, "metrics"):
            self.metrics = metrics
        self.to_render = False
        self.done = False
        self.reset()
        self.global_step = 0

    def close(self):
        pass

    # def execute_action(self, action):
    #     pass

    def reset(self):
        pass

    def render(self):
        pass

    def setRender(self, to_render):
        self.to_render = to_render

    def get_action_space(self):
        pass
