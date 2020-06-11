from rlkit.agents import RandomAgent
from rlkit.environments.gym_environment import GymEnvironment
from rlkit.environments.vizdoom_environment import VizDoomEnvironment
from rlkit.trainers import BasicTrainer

params = {
    "environment_params": {
        # "env_name": "SpaceInvaders-v0",
    },
    "agent_params": {

    },
    "training_params": {
        "run_name": "test_run",
        "train_interval": 10,
        "episodes": 5,
        "steps": 500,
    },
}

# env = GymEnvironment(params["environment_params"])
env = VizDoomEnvironment(params["environment_params"])
agent = RandomAgent(params["agent_params"], env.get_action_space())
trainer = BasicTrainer(params["training_params"], agent, env)
trainer.train()
