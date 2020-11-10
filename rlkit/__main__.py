from rlkit.agents import RandomAgent
from rlkit.environments.gym_environment import GymEnvironment
from rlkit.trainers import BasicTrainer

SEED = 1234
params = dict(
    environment_params = dict(
        env_name = "SpaceInvaders-v0",
        seed = SEED,
    ),
    agent_params = dict(
        seed = SEED,
    ),
    training_params= dict(
        run_name = "test_run",
        train_interval = 10,
        episodes = 5,
        steps = 500,
        seed = SEED,
    ),
)

env = GymEnvironment(params["environment_params"])
agent = RandomAgent(params["agent_params"], env.get_action_space())
trainer = BasicTrainer(params["training_params"], agent, env)
trainer.train()
