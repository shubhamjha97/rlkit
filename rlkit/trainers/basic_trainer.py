from rlkit.core import BaseTrainer
from rlkit.core.metrics import Metrics # TODO: fix import path


class BasicTrainer(BaseTrainer):
    def __init__(self, params, agent, environment, metrics):
        super(BasicTrainer, self).__init__(params, agent, environment, metrics)

        self.train_interval = params["train_interval"]
        self.run_name = params["run_name"]
        self.episodes = params["episodes"]
        self.steps = params["steps"]

    def do_step(self):
        action = self.agent.get_action(self.environment.state)
        return self.environment.step(action)
        # self.environment.render() # TODO: find better solution

    def train(self):
        try:
            for episode in range(1, self.episodes+1):
                step = 0
                self.environment.reset()
                while step < self.steps and not self.environment.done:
                    print("episode: {}, step: {}".format(episode, step))
                    state, reward, done, info = self.do_step()

                    # Train agent
                    if self.global_step > 0 and not self.global_step % self.train_interval:
                        self.agent.train()

                    # Increment step counts
                    step += 1
                    self.global_step += 1

                    self.metrics.log_metric("timestep", step, log_step=self.global_step, namespace="trainer")
                    self.metrics.log_metric("episode", episode, log_step=self.global_step, namespace="trainer")
        finally:
            self.environment.close()
