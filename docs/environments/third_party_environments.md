```{eval-rst}
:tocdepth: 2
```

# Third-Party Environments

The Farama Foundation maintains a number of other [projects](https://farama.org/projects), most of which use Gymnasium. Topics include:
multi-agent RL ([PettingZoo](https://pettingzoo.farama.org/)),
offline-RL ([Minari](https://minari.farama.org/)),
gridworlds ([Minigrid](https://minigrid.farama.org/)),
robotics ([Gymnasium-Robotics](https://robotics.farama.org/)),
multi-objective RL ([MO-Gymnasium](https://mo-gymnasium.farama.org/))
many-agent RL ([MAgent2](https://magent2.farama.org/)),
3D navigation ([Miniworld](https://miniworld.farama.org/)), and many more.

## Third-party environments with Gymnasium

*This page contains environments which are not maintained by Farama Foundation and, as such, cannot be guaranteed to function as intended.*

*If you'd like to contribute an environment, please reach out on [Discord](https://discord.gg/bnJ6kubTg6).*

### [CARL: context adaptive RL](https://github.com/automl/CARL)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.27.1-blue)
![GitHub stars](https://img.shields.io/github/stars/automl/carl)

Contextual extensions of popular reinforcement learning environments that enable training and test distributions for generalization, e.g. CartPole with variable pole lengths or Brax robots with different ground frictions.

### [DACBench: a benchmark for Dynamic Algorithm Configuration](https://github.com/automl/DACBench)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.26.3-blue)
![GitHub stars](https://img.shields.io/github/stars/automl/DACBench)

A benchmark library for [Dynamic Algorithm Configuration](https://www.automl.org/dynamic-algorithm-configuration/). Its focus is on reproducibility and comparability of different DAC methods as well as easy analysis of the optimization process.

### [flappy-bird-env](https://github.com/robertoschiavone/flappy-bird-env)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.28.1-blue)
![GitHub stars](https://img.shields.io/github/stars/robertoschiavone/flappy-bird-env)

Flappy Bird as a Farama Gymnasium environment.

### [flappy-bird-gymnasium: A Flappy Bird environment for Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.27.1-blue)
![GitHub stars](https://img.shields.io/github/stars/markub3327/flappy-bird-gymnasium)

A simple environment for single-agent reinforcement learning algorithms on a clone of [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird), the hugely popular arcade-style mobile game. Both state and pixel observation environments are available.

### [gym-cellular-automata: Cellular Automata environments](https://github.com/elbecerrasoto/gym-cellular-automata)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.28.1-blue)
![GitHub stars](https://img.shields.io/github/stars/elbecerrasoto/gym-cellular-automata)

Environments where the agent interacts with _Cellular Automata_ by changing its cell states.

### [gym-jiminy: Training Robots in Jiminy](https://github.com/duburcqa/jiminy)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.28.0-blue)
![GitHub stars](https://img.shields.io/github/stars/duburcqa/jiminy)

gym-jiminy presents an extension of the initial Gym for robotics using [Jiminy](https://github.com/duburcqa/jiminy), an extremely fast and light-weight simulator for poly-articulated systems using Pinocchio for physics evaluation and Meshcat for web-based 3D rendering.

### [gym-pybullet-drones: Environments for quadcopter control](https://github.com/JacopoPan/gym-pybullet-drones)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.27.1-blue)
![GitHub stars](https://img.shields.io/github/stars/JacopoPan/gym-pybullet-drones)

A simple environment using [PyBullet](https://github.com/bulletphysics/bullet3) to simulate the dynamics of a [Bitcraze Crazyflie 2.x](https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf) nanoquadrotor.

### [gym-saturation: Environments used to prove theorems](https://github.com/inpefess/gym-saturation)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.27.1-blue)
![GitHub stars](https://img.shields.io/github/stars/inpefess/gym-saturation)

An environment for guiding automated theorem provers based on saturation algorithms (e.g. [Vampire](https://github.com/vprover/vampire)).

### [gym-trading-env: Trading Environment](https://gym-trading-env.readthedocs.io/)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.28.1-blue)
![Github stars](https://img.shields.io/github/stars/ClementPerroud/Gym-Trading-Env)

Gym Trading Env simulates stock (or crypto) market from historical data. It was designed to be fast and easily customizable.

### [highway-env: Autonomous driving and tactical decision-making tasks](https://github.com/eleurent/highway-env)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.27.1-blue)
![GitHub stars](https://img.shields.io/github/stars/eleurent/highway-env)

An environment for behavioral planning in autonomous driving, with an emphasis on high-level perception and decision rather than low-level sensing and control.

### [matrix-mdp: Easily create discrete MDPs](https://github.com/Paul-543NA/matrix-mdp-gym)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.26.2-blue)
![GitHub stars](https://img.shields.io/github/stars/Paul-543NA/matrix-mdp-gym)

An environment to easily implement discrete MDPs as gym environments. Turn a set of matrices (`P_0(s)`, `P(s'| s, a)` and `R(s', s, a)`) into a gym environment that represents the discrete MDP ruled by these dynamics.

### [mobile-env: Environments for coordination of wireless mobile networks](https://github.com/stefanbschneider/mobile-env)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.28.1-blue)
![GitHub stars](https://img.shields.io/github/stars/stefanbschneider/mobile-env)

An open, minimalist Gymnasium environment for autonomous coordination in wireless mobile networks.

### [panda-gym: Robotics environments using the PyBullet physics engine](https://github.com/qgallouedec/panda-gym/)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.26.3-blue)
![GitHub stars](https://img.shields.io/github/stars/qgallouedec/panda-gym)

PyBullet based simulations of a robotic arm moving objects.

### [pystk2-gymnasium: SuperTuxKart races gymnasium wrapper](https://github.com/bpiwowar/pystk2-gymnasium)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.29.1-blue)
![GitHub stars](https://img.shields.io/github/stars/bpiwowar/pystk2-gymnasium)

Uses a [python wrapper](https://github.com/bpiwowar/pystk2) around [SuperTuxKart](https://supertuxkart.net/fr/Main_Page) that allows to access the world state and to control a race.

### [QWOP: An environment for Bennet Foddy's game QWOP](https://github.com/smanolloff/qwop-gym)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.29.1-blue)
![GitHub stars](https://img.shields.io/github/stars/smanolloff/qwop-gym)

QWOP is a game about running extremely fast down a 100 meter track. With this Gymnasium environment you can train your own agents and try to beat the current world record (5.0 in-game seconds for humans and 4.7 for AI).

### [Safety-Gymnasium: Ensuring safety in real-world RL scenarios](https://github.com/PKU-MARL/safety-gymnasium)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.28.1-blue)
![GitHub stars](https://img.shields.io/github/stars/PKU-MARL/safety-gymnasium)

Highly scalable and customizable Safe Reinforcement Learning library.

### [SimpleGrid: a simple grid environment for Gymnasium](https://github.com/damat-le/gym-simplegrid)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.28.1-blue)
![GitHub stars](https://img.shields.io/github/stars/damat-le/gym-simplegrid)

SimpleGrid is a super simple and minimal grid environment for Gymnasium. It is easy to use and customise and it is intended to offer an environment for rapidly testing and prototyping different RL algorithms.

### [spark-sched-sim: Environments for scheduling DAG jobs in Apache Spark](https://github.com/ArchieGertsman/spark-sched-sim)

spark-sched-sim simulates Spark clusters for RL-based job scheduling algorithms. Spark jobs are encoded as directed acyclic graphs (DAGs), providing opportunities to experiment with graph neural networks (GNN's) in the RL context.

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.29.1-blue)
![GitHub stars](https://img.shields.io/github/stars/ArchieGertsman/spark-sched-sim)

### [stable-retro: Classic retro games, a maintained version of OpenAI Retro](https://github.com/Farama-Foundation/stable-retro)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.27.1-blue)
![GitHub stars](https://img.shields.io/github/stars/Farama-Foundation/stable-retro)

Supported fork of [gym-retro](https://openai.com/research/gym-retro): turn classic video games into Gymnasium environments.

### [sumo-rl: Reinforcement Learning using SUMO traffic simulator](https://github.com/LucasAlegre/sumo-rl)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.26.3-blue)
![GitHub stars](https://img.shields.io/github/stars/LucasAlegre/sumo-rl)

Gymnasium wrapper for various environments in the SUMO traffic simulator. Supports both single and multiagent settings (using [pettingzoo](https://pettingzoo.farama.org/)).

### [tmrl: TrackMania 2020 through RL](https://github.com/trackmania-rl/tmrl/)

![Gymnasium version dependency](https://img.shields.io/badge/Gymnasium-v0.27.1-blue)
![GitHub stars](https://img.shields.io/github/stars/trackmania-rl/tmrl)

tmrl is a distributed framework for training Deep Reinforcement Learning AIs in real-time applications. It is demonstrated on the TrackMania 2020 video game.

## Third-Party Environments using Gym

There are a large number of third-party environments using various versions of [Gym](https://github.com/openai/gym).
Many of these can be adapted to work with gymnasium (see [Compatibility with Gym](https://gymnasium.farama.org/content/gym_compatibility/)), but are not guaranteed to be fully functional.

## Video Game environments

### [gym-derk: GPU accelerated MOBA environment](https://gym.derkgame.com/)

A 3v3 MOBA environment where you train creatures to fight each other.

### [SlimeVolleyGym: A simple environment for Slime Volleyball game](https://github.com/hardmaru/slimevolleygym)

A simple environment for benchmarking single and multi-agent reinforcement learning algorithms on a clone of Slime Volleyball game.

### [Unity ML Agents: Environments for Unity game engine](https://github.com/Unity-Technologies/ml-agents)

Gym (and PettingZoo) wrappers for arbitrary and premade environments with the Unity game engine.

### [PGE: Parallel Game Engine](https://github.com/222464/PGE)

Uses The [Open 3D Engine](https://www.o3de.org/) for AI simulations and can interoperate with the Gym. Uses [PyBullet](https://github.com/bulletphysics/bullet3) physics.

## Robotics environments

### [PyFlyt: UAV Flight Simulator Environments for Reinforcement Learning Research](https://jjshoots.github.io/PyFlyt/index.html#)

A library for testing reinforcement learning algorithms on various UAVs.
It is built on the [Bullet](https://github.com/bulletphysics/bullet3) physics engine, offers flexible rendering options, time-discrete steppable physics, Python bindings, and support for custom drones of any configuration, be it biplanes, quadcopters, rockets, and anything you can think of.

### [MarsExplorer: Environments for controlling robot on Mars](https://github.com/dimikout3/MarsExplorer)

Mars Explorer is a Gym compatible environment designed and developed as an initial endeavor to bridge the gap between powerful Deep Reinforcement Learning methodologies and the problem of exploration/coverage of an unknown terrain.

### [robo-gym: Real-world and simulation robotics](https://github.com/jr-robotics/robo-gym)

Robo-gym provides a collection of reinforcement learning environments involving robotic tasks applicable in both simulation and real-world robotics.

### [Offworld-gym: Control real robots remotely for free](https://github.com/offworld-projects/offworld-gym)

Gym environments that let you control real robots in a laboratory via the internet.

### [safe-control-gym: Evaluate safety of RL algorithms](https://github.com/utiasDSL/safe-control-gym)

Evaluate safety, robustness and generalization via PyBullet based CartPole and Quadrotor environments—with [CasADi](https://web.casadi.org) (symbolic) *a priori* dynamics and constraints.

### [gym-softrobot: Soft-robotics environments](https://github.com/skim0119/gym-softrobot/)

A large-scale benchmark for co-optimizing the design and control of soft robots.

### [iGibson: Photorealistic and interactive robotics environments](https://github.com/StanfordVL/iGibson/)

A simulation environment with high-quality realistic scenes, with interactive physics using [PyBullet](https://github.com/bulletphysics/bullet3).

### [DexterousHands: Dual dexterous hand manipulation tasks](https://github.com/PKU-MARL/DexterousHands/)

This is a library that provides dual dexterous hand manipulation tasks through Isaac Gym.

### [OmniIsaacGymEnvs: Gym environments for NVIDIA Omniverse Isaac ](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/)

Reinforcement Learning Environments for [Omniverse Isaac simulator](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html).

## Autonomous Driving environments

### [gym-duckietown: Lane-following for self-driving cars](https://github.com/duckietown/gym-duckietown)

A lane-following simulator built for the [Duckietown](http://duckietown.org/) project (small-scale self-driving car course).

### [gym-electric-motor: Gym environments for electric motor simulations](https://github.com/upb-lea/gym-electric-motor)

An environment for simulating a wide variety of electric drives taking into account different types of electric motors and converters.

### [CommonRoad-RL: Motion planning for traffic scenarios ](https://commonroad.in.tum.de/tools/commonroad-rl)

A Gym for solving motion planning problems for various traffic scenarios compatible with [CommonRoad benchmarks](https://commonroad.in.tum.de/scenarios), which provides configurable rewards, action spaces, and observation spaces.

### [racing_dreamer: Latent imagination in autonomous racing](https://github.com/CPS-TUWien/racing_dreamer/)

Train a model-based RL agent in simulation and, without finetuning, transfer it to small-scale race cars.

### [l2r: Multimodal control environment where agents learn how to race](https://github.com/learn-to-race/l2r/)

An open-source reinforcement learning environment for autonomous racing.

### [racecar_gym: Miniature racecar env using PyBullet](https://github.com/axelbr/racecar_gym/)

A gym environment for a miniature racecar using the [PyBullet](https://github.com/bulletphysics/bullet3) physics engine.

## Other environments

### [Connect-4-gym : An environment for practicing self playing](https://github.com/lucasBertola/Connect-4-Gym-env-Reinforcement-learning)

Connect-4-Gym is an environment designed for creating AIs that learn by playing against themselves and assigning them an Elo rating. This environment can be used to train and evaluate reinforcement learning agents on the classic board game Connect Four.

### [CompilerGym: Optimise compiler tasks](https://github.com/facebookresearch/CompilerGym)

Reinforcement learning environments for compiler optimization tasks, such as LLVM phase ordering, GCC flag tuning, and CUDA loop nest code generation.

### [gym-sokoban: 2D Transportation Puzzles](https://github.com/mpSchrader/gym-sokoban)

The environment consists of transportation puzzles in which the player's goal is to push all boxes to the warehouse's storage locations.

### [NLPGym: A toolkit to develop RL agents to solve NLP tasks](https://github.com/rajcscw/nlp-gym)

[NLPGym](https://arxiv.org/pdf/2011.08272v1.pdf) provides interactive environments for standard NLP tasks such as sequence tagging, question answering, and sequence classification.

### [ShinRL: Environments for evaluating RL algorithms](https://github.com/omron-sinicx/ShinRL/)

ShinRL: A Library for Evaluating RL Algorithms from Theoretical and Practical Perspectives (Deep RL Workshop 2021)

### [gymnax: Hardware Accelerated RL Environments](https://github.com/RobertTLange/gymnax/)

RL Environments in JAX which allows for highly vectorised environments with support for a number of environments, Gym, MinAtari, bsuite and more.

### [gym-anytrading: Financial trading environments for FOREX and STOCKS](https://github.com/AminHP/gym-anytrading)

AnyTrading is a collection of Gym environments for reinforcement learning-based trading algorithms with a great focus on simplicity, flexibility, and comprehensiveness.

### [gym-mtsim: Financial trading for MetaTrader 5 platform](https://github.com/AminHP/gym-mtsim)

MtSim is a simulator for the [MetaTrader 5](https://www.metatrader5.com/) trading platform for reinforcement learning-based trading algorithms.

### [openmodelica-microgrid-gym: Environments for controlling power electronic converters in microgrids](https://github.com/upb-lea/openmodelica-microgrid-gym)

The OpenModelica Microgrid Gym (OMG) package is a software toolbox for the simulation and control optimization of microgrids based on energy conversion by power electronic converters.

### [GymFC: A flight control tuning and training framework](https://github.com/wil3/gymfc/)

GymFC is a modular framework for synthesizing neuro-flight controllers. Has been used to generate policies for the world's first open-source neural network flight control firmware [Neuroflight](https://github.com/wil3/neuroflight).
