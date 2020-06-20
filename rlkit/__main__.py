from rlkit.agents import RandomAgent
from rlkit.core import Metrics
from rlkit.environments.gym_environment import GymEnvironment
from rlkit.trainers import BasicTrainer

params = {

    "experiment_params": {
        "experiment_name": "debug_expt-0"
    }

    "environment_params": {
        "env_name": "SpaceInvaders-v0",
    },
    "agent_params": {},
    "training_params": {
        "run_name": "test_run",
        "train_interval": 10,
        "episodes": 5,
        "steps": 500,
    },
}

metrics = Metrics()
env = GymEnvironment(params["environment_params"], metrics)
agent = RandomAgent(params["agent_params"], env.get_action_space(), metrics)
trainer = BasicTrainer(params["training_params"], agent, env, metrics)
trainer.train()
