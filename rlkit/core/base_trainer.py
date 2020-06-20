from collections import defaultdict


class BaseTrainer:
    def __init__(self, params, agent, environment, metrics):
        self.metrics = metrics
        self.agent = agent
        self.environment = environment

        self.global_step = 0

    def do_step(self):
        pass

    def train(self):
        pass

