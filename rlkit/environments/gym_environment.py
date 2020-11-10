import gym
from rlkit.core import BaseEnvironment


class GymEnvironment(BaseEnvironment):
    def __init__(self, params):
        self.params = params
        self.env_name = params["env_name"]
        self.env = gym.make(self.env_name)
        super(GymEnvironment, self).__init__()

    def execute_action(self, action):
        self.env.step(action)

    def get_action_space(self):
        return self.env.action_space

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
        self.env.reset()

    def close(self):
        print("closing env")
        return self.env.close()

    def render(self):
        self.env.render()

    def step(self, action):
        self.state, self.reward, self.done, self.info = self.env.step(action)
        return (self.state, self.reward, self.done, self.info, )


if __name__ == "__main__":
    test_env = GymEnvironment("MountainCarContinuous-v0")
