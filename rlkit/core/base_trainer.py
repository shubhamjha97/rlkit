class BaseTrainer:
    def __init__(self, params):
        self.global_step = 0
        self.episodes = params.get("episodes", 10);
        self.steps = params.get("steps", 100)

    def do_step(self):
        pass

    def train(self):
        pass
