# RLkit: A simple Reinforcement Learning library

[![PyPI version shields.io](https://img.shields.io/badge/pypi-0.2.0-brightgreen.svg?style=flat-square)](https://pypi.org/project/RLkit/) [![PyPI license](https://img.shields.io/apm/l/vim-mode.svg?style=flat-square)](https://img.shields.io/apm/l/vim-mode.svg?style=flat-square)

This project is still a work in progress. More algorithms and detailed documentation coming soon :)

Currently supported agents-

1. Random agent
2. REINFORCE (Policy Gradients)
3. DQN
4. DQN with baseline
5. Actor-Critic

See examples for details on how to use the library.

## Installation:

#### Stable:

```shell
$ pip install -U RLkit
```

#### Dev:

To get the project's source code, clone the github repository:

```shell
$ git clone https://github.com/shubhamjha97/rlkit.git
$ cd RLkit
```

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv venv
$ source venv/bin/activate
```

Install all the required packages:

```shell
$ pip install -r requirements.txt
```

Install the package by running the following command from the root directory of the repository:

```shell
$ python setup.py install	
```

## Requirements-
```
tensorflow==1.11.0
gym==0.10.8
numpy==1.15.4
```

## New in v0.2
- Added DQN and DQN with baseline agents
- Added ActorCritic agent
- Added support for various activation functions


## Upcoming
- Duelling DQN
- Support for logging and plotting
- Support for adding seeds
- Support for custom environments

### Compatibility

The package has been tested with python 3.5.2


## References

- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602. [[paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)]

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Petersen, S. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529. [[paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)]

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4), 229-256. [[paper](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)]

- Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-critic algorithms. In Advances in neural information processing systems (pp. 1008-1014). [[paper](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)]