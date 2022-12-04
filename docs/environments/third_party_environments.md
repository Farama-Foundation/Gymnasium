```{eval-rst}
:tocdepth: 2
```

# Third-party Environments

There are a number of Reinforcement Learning environments built by authors not included with Gymnasium. The Farama Foundation maintains a number of projects for gridworlds, procedurally generated worlds, video games, robotics, these can be found at [projects](https://farama.org/projects).

## Video Game environments

### [stable-retro: Classic retro games, a maintained version of OpenAI Retro](https://github.com/MatPoliquin/stable-retro)

Supported fork of gym-retro with additional games, states, scenarios, etc. Open to PRs of additional games, features, and platforms since gym-retro is no longer maintained

### [flappy-bird-gym: A Flappy Bird environment for Gym](https://github.com/Talendar/flappy-bird-gym)

A simple environment for single-agent reinforcement learning algorithms on a clone of [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird), the hugely popular arcade-style mobile game. Both state and pixel observation environments are available.

### [gym-derk: GPU accelerated MOBA environment](https://gym.derkgame.com/)

This is a 3v3 MOBA environment where you train creatures to fight each other. It runs entirely on the GPU so you can easily have hundreds of instances running in parallel. There are around 15 items for the creatures, 60 "senses", 5 actions, and roughly 23 tweakable rewards. It's also possible to benchmark an agent against other agents online. It's available for free for training for personal use, and otherwise costs money; see licensing details on the website

### [SlimeVolleyGym: A simple environment for single and multi-agent reinforcement learning](https://github.com/hardmaru/slimevolleygym)

A simple environment for benchmarking single and multi-agent reinforcement learning algorithms on a clone of Slime Volleyball game. The only dependencies are gym and NumPy. Both state and pixel observation environments are available. The motivation of this environment is to easily enable trained agents to play against each other, and also facilitate the training of agents directly in a multi-agent setting, thus adding an extra dimension for evaluating an agent's performance.

### [Unity ML Agents: Environments for Unity game engine](https://github.com/Unity-Technologies/ml-agents)

Gym wrappers for arbitrary and premade environments with the Unity game engine.

### [PGE: Parallel Game Engine](https://github.com/222464/PGE)

PGE is a FOSS 3D engine for AI simulations and can interoperate with the Gym. Contains environments with modern 3D graphics, and uses Bullet for physics.

## Robotics environments

### [gym-jiminy: Training Robots in Jiminy](https://github.com/duburcqa/jiminy)

gym-jiminy presents an extension of the initial Gym for robotics using Jiminy, an extremely fast and light-weight simulator for poly-articulated systems using Pinocchio for physics evaluation and Meshcat for web-based 3D rendering.

### [gym-pybullet-drones: Environments for quadcopter control](https://github.com/JacopoPan/gym-pybullet-drones)

A simple environment using [PyBullet](https://github.com/bulletphysics/bullet3) to simulate the dynamics of a [Bitcraze Crazyflie 2.x](https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf) nanoquadrotor.

### [MarsExplorer: Environments for controlling robot on Mars](https://github.com/dimikout3/MarsExplorer)

Mars Explorer is a Gym compatible environment designed and developed as an initial endeavor to bridge the gap between powerful Deep Reinforcement Learning methodologies and the problem of exploration/coverage of an unknown terrain.

### [panda-gym: Robotics environments using the PyBullet physics engine](https://github.com/qgallouedec/panda-gym/)

PyBullet based simulations of a robotic arm moving objects.

### [robo-gym: Real-world and simulation robotics](https://github.com/jr-robotics/robo-gym)

Robo-gym provides a collection of reinforcement learning environments involving robotic tasks applicable in both simulation and real-world robotics.

### [Offworld-gym](https://github.com/offworld-projects/offworld-gym)

Gym environments that let you control physics robotics in a laboratory via the internet.

### [safe-control-gym](https://github.com/utiasDSL/safe-control-gym)

PyBullet based CartPole and Quadrotor environments—with [CasADi](https://web.casadi.org) (symbolic) *a priori* dynamics and constraints—for learning-based control and model-based reinforcement learning.

### [gym-softrobot: Soft-robotics environments](https://github.com/skim0119/gym-softrobot/)

A large-scale benchmark for co-optimizing the design and control of soft robots.

### [iGibson](https://github.com/StanfordVL/iGibson/)

A Simulation Environment to train Robots in Large Realistic Interactive Scenes

### [DexterousHands: dual dexterous hand manipulation tasks](https://github.com/PKU-MARL/DexterousHands/)

This is a library that provides dual dexterous hand manipulation tasks through Isaac Gym

### [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/)

Reinforcement Learning Environments for Omniverse Isaac Gym

## Autonomous Driving environments

### [sumo-rl](https://github.com/LucasAlegre/sumo-rl)

Gym wrapper for various environments in the Sumo traffic simulator

### [gym-duckietown](https://github.com/duckietown/gym-duckietown)

A lane-following simulator built for the [Duckietown](http://duckietown.org/) project (small-scale self-driving car course).

### [gym-electric-motor](https://github.com/upb-lea/gym-electric-motor)

An environment for simulating a wide variety of electric drives taking into account different types of electric motors and converters. Control schemes can be continuous, yielding a voltage duty cycle, or discrete, determining converter switching states directly.

### [highway-env](https://github.com/eleurent/highway-env)

An environment for behavioral planning in autonomous driving, with an emphasis on high-level perception and decision rather than low-level sensing and control. The difficulty of the task lies in understanding the social interactions with other drivers, whose behaviors are uncertain. Several scenes are proposed, such as highway, merge, intersection and roundabout.

### [CommonRoad-RL](https://commonroad.in.tum.de/tools/commonroad-rl)

A Gym for solving motion planning problems for various traffic scenarios compatible with [CommonRoad benchmarks](https://commonroad.in.tum.de/scenarios), which provides configurable rewards, action spaces, and observation spaces.

### [tmrl: TrackMania 2020 through RL](https://github.com/trackmania-rl/tmrl/)

tmrl is a distributed framework for training Deep Reinforcement Learning AIs in real-time applications. It is demonstrated on the TrackMania 2020 video game.

### [racing_dreamer](https://github.com/CPS-TUWien/racing_dreamer/)

Latent Imagination Facilitates Zero-Shot Transfer in Autonomous Racing

### [l2r: Multimodal control environment where agents learn how to race](https://github.com/learn-to-race/l2r/)

An open-source reinforcement learning environment for autonomous racing.

### [racecar_gym](https://github.com/axelbr/racecar_gym/)

A gym environment for a miniature racecar using the pybullet physics engine.

## Other environments

### [CompilerGym: Optimise compiler tasks](https://github.com/facebookresearch/CompilerGym)

Reinforcement learning environments for compiler optimization tasks, such as LLVM phase ordering, GCC flag tuning, and CUDA loop nest code generation.

### [CARL: context adaptive RL](https://github.com/automl/CARL)

Configurable reinforcement learning environments for testing generalization, e.g. CartPole with variable pole lengths or Brax robots with different ground frictions.

### [matrix-mdp: Easily create discrete MDPs](https://github.com/Paul-543NA/matrix-mdp-gym)

An environment to easily implement discrete MDPs as gym environments. Turn a set of matrices (`P_0(s)`, `P(s'| s, a)` and `R(s', s, a)`) into a gym environment that represents the discrete MDP ruled by these dynamics.

### [mo-gym: Multi-objective Reinforcement Learning environments](https://github.com/LucasAlegre/mo-gym)

Multi-objective RL (MORL) gym environments, where the reward is a NumPy array of different (possibly conflicting) objectives.

### [gym-cellular-automata: Cellular Automata environments](https://github.com/elbecerrasoto/gym-cellular-automata)

Environments where the agent interacts with _Cellular Automata_ by changing its cell states.

### [gym-sokoban: 2D Transportation Puzzles](https://github.com/mpSchrader/gym-sokoban)

The environment consists of transportation puzzles in which the player's goal is to push all boxes to the warehouse's storage locations. The advantage of the environment is that it generates a new random level every time it is initialized or reset, which prevents overfitting to predefined levels.

### [DACBench: Benchmark Library for Dynamic Algorithm configuration](https://github.com/automl/DACBench)

Environments for hyperparameter configuration using RL. Includes cheap surrogate benchmarks as well as real-world algorithms from e.g. AI Planning, Evolutionary Computation and Deep Learning.

### [NLPGym: A toolkit to develop RL agents to solve NLP tasks](https://github.com/rajcscw/nlp-gym)

[NLPGym](https://arxiv.org/pdf/2011.08272v1.pdf) provides interactive environments for standard NLP tasks such as sequence tagging, question answering, and sequence classification. Users can easily customize the tasks with their datasets, observations, features and reward functions.

### [gym-saturation: Environments used to prove theorems](https://github.com/inpefess/gym-saturation)

An environment for guiding automated theorem provers based on saturation algorithms (e.g. [Vampire](https://github.com/vprover/vampire)).

### [ShinRL: Environments for evaluating RL algorithms](https://github.com/omron-sinicx/ShinRL/)

ShinRL: A Library for Evaluating RL Algorithms from Theoretical and Practical Perspectives (Deep RL Workshop 2021)

### [gymnax: Hardware Accelerated RL Environments](https://github.com/RobertTLange/gymnax/)

RL Environments in JAX which allows for highly vectorised environments with support for a number of environments, Gym, MinAtari, bsuite and more.

### [gym-anytrading: Financial trading environments for FOREX and STOCKS](https://github.com/AminHP/gym-anytrading)

AnyTrading is a collection of Gym environments for reinforcement learning-based trading algorithms with a great focus on simplicity, flexibility, and comprehensiveness.

### [gym-mtsim: Financial trading for MetaTrader 5 platform](https://github.com/AminHP/gym-mtsim)

MtSim is a simulator for the MetaTrader 5 trading platform for reinforcement learning-based trading algorithms. MetaTrader 5 is a multi-asset platform that allows trading Forex, Stocks, Crypto, and Futures.

### [openmodelica-microgrid-gym: Environments for controlling power electronic converters in microgrids](https://github.com/upb-lea/openmodelica-microgrid-gym)

The OpenModelica Microgrid Gym (OMG) package is a software toolbox for the simulation and control optimization of microgrids based on energy conversion by power electronic converters.

### [mobile-env: Environments for coordination of wireless mobile networks](https://github.com/stefanbschneider/mobile-env/)

An open, minimalist Gym environment for autonomous coordination in wireless mobile networks.

### [GymFC: A flight control tuning and training framework](https://github.com/wil3/gymfc/)

GymFC is a modular framework for synthesizing neuro-flight controllers. The architecture integrates digital twinning concepts to provide a seamless transfer of trained policies to hardware. The environment has been used to generate policies for the world's first open-source neural network flight control firmware [Neuroflight](https://github.com/wil3/neuroflight).
