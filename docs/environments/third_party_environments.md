# Third-Party Environments

Many environments that comply with the Gymnasium API are now maintained under the Farama Foundation's [projects](https://farama.org/projects), along with Gymnasium itself. These include many of the most popular environments using the Gymnasium API, and we encourage you to check them out. This page exclusively lists interesting third party environments that are not part of the Farama Foundation.


## Video Game Environments

### [ flappy-bird-gym: A Flappy Bird environment for Gym](https://github.com/Talendar/flappy-bird-gym)

A simple environment for single-agent reinforcement learning algorithms on a clone of [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird), the hugely popular arcade-style mobile game. Both state and pixel observation environments are available.

### [ gym-derk: GPU accelerated MOBA environment](https://gymnasium.derkgame.com)

This is a 3v3 MOBA environment where you train creatures to fight each other. It runs entirely on the GPU so you can easily have hundreds of instances running in parallel. There are around 15 items for the creatures, 60 "senses", 5 actions, and roughly 23 tweakable rewards. It's also possible to benchmark an agent against other agents online. It's available for free for training for personal use, and otherwise costs money; see licensing details on the website

### [ SlimeVolleyGym: A simple environment for single and multi-agent reinforcement learning](https://github.com/hardmaru/slimevolleygym)

A simple environment for benchmarking single and multi-agent reinforcement learning algorithms on a clone of Slime Volleyball game. The only dependencies are gym and NumPy. Both state and pixel observation environments are available. The motivation of this environment is to easily enable trained agents to play against each other, and also facilitate the training of agents directly in a multi-agent setting, thus adding an extra dimension for evaluating an agent's performance.

### [ stable-retro](https://github.com/MatPoliquin/stable-retro)

Supported fork of gym-retro with additional games, states, scenarios, etc. Open to PRs of additional games, features, and platforms since gym-retro is no longer maintained

### [ Unity ML Agents](https://github.com/Unity-Technologies/ml-agents)

Gym wrappers for arbitrary and premade environments with the Unity game engine.

### [ gym-games](https://github.com/qlan3/gym-games)

Gym implementations of the MinAtar games, various PyGame Learning Environment games, and various custom exploration games

### [ PGE: Parallel Game Engine](https://github.com/222464/PGE)

PGE is a FOSS 3D engine for AI simulations and can interoperate with the Gym. Contains environments with modern 3D graphics, and uses Bullet for physics.


## Robotics Environments

### [GymFC: A flight control tuning and training framework](https://github.com/wil3/gymfc/)

GymFC is a modular framework for synthesizing neuro-flight controllers. The architecture integrates digital twinning concepts to provide a seamless transfer of trained policies to hardware. The environment has been used to generate policies for the world's first open-source neural network flight control firmware [Neuroflight](https://github.com/wil3/neuroflight).

### [gym-gazebo](https://github.com/erlerobot/gym-gazebo/)

gym-gazebo presents an extension of the initial Gym for robotics using ROS and Gazebo, an advanced 3D modeling and
rendering tool.

### [gym-goddard: Goddard's Rocket Problem](https://github.com/osannolik/gym-goddard)

An environment for simulating the classical optimal control problem where the thrust of a vertically ascending rocket shall be determined such that it reaches the maximum possible altitude while being subject to varying aerodynamic drag, gravity, and mass.

### [gym-jiminy: training Robots in Jiminy](https://github.com/Wandercraft/jiminy)

gym-jiminy presents an extension of the initial Gym for robotics using Jiminy, an extremely fast and light-weight simulator for poly-articulated systems using Pinocchio for physics evaluation and Meshcat for web-based 3D rendering.

### [gym-pybullet-drones](https://github.com/JacopoPan/gym-pybullet-drones)

A simple environment using [PyBullet](https://github.com/bulletphysics/bullet3) to simulate the dynamics of a [Bitcraze Crazyflie 2.x](https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf) nanoquadrotor.

### [MarsExplorer](https://github.com/dimikout3/MarsExplorer)

Mars Explorer is a Ggym compatible environment designed and developed as an initial endeavor to bridge the gap between powerful Deep Reinforcement Learning methodologies and the problem of exploration/coverage of an unknown terrain.

### [panda-gym ](https://github.com/qgallouedec/panda-gym/)

PyBullet based simulations of a robotic arm moving objects.

### [robo-gym](https://github.com/jr-robotics/robo-gym)

robo-gym provides a collection of reinforcement learning environments involving robotic tasks applicable in both simulation and real-world robotics.

### [Offworld-gym](https://github.com/offworld-projects/offworld-gym)

Gym environments that let you control physics robotics in a laboratory via the internet.

### [osim-rl](https://github.com/stanfordnmbl/osim-rl)

Musculoskeletal Models in OpenSim. A human musculoskeletal model and a physics-based simulation environment where you can synthesize physically and physiologically accurate motion. One of the environments built in this framework is a competition environment for a NIPS 2017 challenge.

### [safe-control-gym](https://github.com/utiasDSL/safe-control-gym)

PyBullet based CartPole and Quadrotor environments—with [CasADi](https://web.casadi.org) (symbolic) *a priori* dynamics and constraints—for learning-based control and model-based reinforcement learning.

### [racecar_gym](https://github.com/axelbr/racecar_gym/)

A gym environment for a miniature racecar using the pybullet physics engine.

### [jiminy](https://github.com/duburcqa/jiminy/)

Jiminy: a fast and portable Python/C++ simulator of poly-articulated systems with Gym interface for reinforcement learning

### [gym-softrobot](https://github.com/skim0119/gym-softrobot/)

Softrobotics environment package for Gym

### [ostrichrl](https://github.com/vittorione94/ostrichrl/)

This is the repository accompanying the paper [OstrichRL: A Musculoskeletal Ostrich Simulation to Study Bio-mechanical Locomotion](https://arxiv.org/abs/2112.06061).

### [quadruped-gym](https://github.com/dtch1997/quadruped-gym/)

A Gym environment for the training of legged robots

### [evogym](https://github.com/EvolutionGym/evogym/)

A large-scale benchmark for co-optimizing the design and control of soft robots, as seen in NeurIPS 2021.

### [iGibson](https://github.com/StanfordVL/iGibson/)

A Simulation Environment to train Robots in Large Realistic Interactive Scenes

### [DexterousHands](https://github.com/PKU-MARL/DexterousHands/)

This is a library that provides dual dexterous hand manipulation tasks through Isaac Gym

### [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/)

Reinforcement Learning Environments for Omniverse Isaac Gym

### [SpaceRobotEnv](https://github.com/Tsinghua-Space-Robot-Learning-Group/SpaceRobotEnv/)

A gym environment designed for free-floating space robot control based on the MuJoCo platform.

### [gym-line-follower](https://github.com/nplan/gym-line-follower/)

Line follower robot simulator environment for Open AI Gym.


## Autonomous Driving and Traffic Control Environments

### [ gym-carla](https://github.com/cjy1992/gym-carla)

gym-carla provides a gym wrapper for the [CARLA simulator](http://carla.org/), which is a realistic 3D simulator for autonomous driving research. The environment includes a virtual city with several surrounding vehicles running around. Multiple sources of observations are provided for the ego vehicle, such as front-view camera image, lidar point cloud image, and bird-eye view semantic mask. Several applications have been developed based on this wrapper, such as deep reinforcement learning for end-to-end autonomous driving.

### [ gym-duckietown](https://github.com/duckietown/gym-duckietown)

A lane-following simulator built for the [Duckietown](http://duckietown.org/) project (small-scale self-driving car course).

### [ gym-electric-motor](https://github.com/upb-lea/gym-electric-motor)

An environment for simulating a wide variety of electric drives taking into account different types of electric motors and converters. Control schemes can be continuous, yielding a voltage duty cycle, or discrete, determining converter switching states directly.

### [ highway-env](https://github.com/eleurent/highway-env)

An environment for behavioral planning in autonomous driving, with an emphasis on high-level perception and decision rather than low-level sensing and control. The difficulty of the task lies in understanding the social interactions with other drivers, whose behaviors are uncertain. Several scenes are proposed, such as highway, merge, intersection and roundabout.

### [ LongiControl](https://github.com/dynamik1703/gym_longicontrol)

An environment for the stochastic longitudinal control of an electric vehicle. It is intended to be a descriptive and comprehensible example of a continuous real-world problem within the field of autonomous driving.

### [ sumo-rl](https://github.com/LucasAlegre/sumo-rl)

Gym wrapper for various environments in the Sumo traffic simulator

### [ CommonRoad-RL](https://commonroad.in.tum.de/tools/commonroad-rl)

A Gym for solving motion planning problems for various traffic scenarios compatible with [CommonRoad benchmarks](https://commonroad.in.tum.de/scenarios), which provides configurable rewards, action spaces, and observation spaces.

### [tmrl](https://github.com/trackmania-rl/tmrl/)

TrackMania 2020 through RL

### [racing_dreamer](https://github.com/CPS-TUWien/racing_dreamer/)

Latent Imagination Facilitates Zero-Shot Transfer in Autonomous Racing

### [l2r](https://github.com/learn-to-race/l2r/)

An open-source reinforcement learning environment for autonomous racing.

### [gym_torcs](https://github.com/ugo-nama-kun/gym_torcs/)

Gym-TORCS is the reinforcement learning (RL) environment in TORCS domain with gym-like interface. TORCS is the open-source realistic car racing simulator recently used as an RL benchmark task in several AI studies.


## Recommendation System Environments

### [ gym-adserve](https://github.com/falox/gym-adserver)

An environment that implements a typical [multi-armed bandit scenario](https://en.wikipedia.org/wiki/Multi-armed_bandit) where an [ad server](https://en.wikipedia.org/wiki/Ad_serving) must select the best advertisement to be displayed in a web page. Some example agents included: Random, epsilon-Greedy, Softmax, and UCB1.

### [ gym-recsys](https://github.com/zuoxingdong/gym-recsys)

This package describes an Gym interface for creating a simulation environment of reinforcement learning-based recommender systems (RL-RecSys). The design strives for simple and flexible APIs to support novel research.

### [ VirtualTaobao](https://github.com/eyounx/VirtualTaobao/)

An environment for online recommendations, where customers are learned from Taobao.com, one of the world's largest e-commerce platforms.


## Industrial Process Environments

### [ gym-inventory](https://github.com/paulhendricks/gym-inventory)

gym-inventory is a single-agent domain featuring discrete state and action spaces that an AI agent might encounter in inventory control problems.


### [ openmodelica-microgrid-gym](https://github.com/upb-lea/openmodelica-microgrid-gym)

The OpenModelica Microgrid Gym (OMG) package is a software toolbox for the simulation and control optimization of microgrids based on energy conversion by power electronic converters.


### [mobile-env](https://github.com/stefanbschneider/mobile-env/)

An open, minimalist Gym environment for autonomous coordination in wireless mobile networks.

### [PyElastica](https://github.com/GazzolaLab/PyElastica/)

Python implementation of Elastica, open-source software for the simulation of assemblies of slender, one-dimensional structures using Cosserat Rod theory.


## Financial Environments


### [ gym-anytrading](https://github.com/AminHP/gym-anytrading)

AnyTrading is a collection of Gym environments for reinforcement learning-based trading algorithms with a great focus on simplicity, flexibility, and comprehensiveness.

### [ gym-mtsim](https://github.com/AminHP/gym-mtsim)

MtSim is a general-purpose, flexible, an



## Other Environments

### [ CARL](https://github.com/automl/CARL)

Configurable reinforcement learning environments for testing generalization, e.g. CartPole with variable pole lengths or Brax robots with different ground frictions.

### [ CompilerGym](https://github.com/facebookresearch/CompilerGym)

Reinforcement learning environments for compiler optimization tasks, such as LLVM phase ordering, GCC flag tuning, and CUDA loop nest code generation.

### [ DACBench](https://github.com/automl/DACBench)

Environments for hyperparameter configuration using RL. Includes cheap surrogate benchmarks as well as real-world algorithms from e.g. AI Planning, Evolutionary Computation and Deep Learning. 

### [ gym-autokey](https://github.com/Flunzmas/gym-autokey)

An environment for automated rule-based deductive program verification in the KeY verification system.

### [ gym-cellular-automata](https://github.com/elbecerrasoto/gym-cellular-automata)

Environments where the agent interacts with _Cellular Automata_ by changing its cell states.

### [ gym-maze](https://github.com/tuzzer/gym-maze/)

A simple 2D maze environment where an agent finds its way from the start position to the goal.

d easy-to-use simulator alongside a Gym trading environment for MetaTrader 5 trading platform.

### [ gym-riverswim](https://github.com/erfanMhi/gym-riverswim)

A simple environment for benchmarking reinforcement learning exploration techniques in a simplified setting. Hard exploration.

### [ gym-sokoban](https://github.com/mpSchrader/gym-sokoban)

2D Transportation Puzzles. The environment consists of transportation puzzles in which the player's goal is to push all boxes to the warehouse's storage locations. The advantage of the environment is that it generates a new random level every time it is initialized or reset, which prevents overfitting to predefined levels.

### [ math-prog-synth-env](https://github.com/JohnnyYeeee/math_prog_synth_env)

In our paper "A Reinforcement Learning Environment for Mathematical Reasoning via Program Synthesis" we convert the DeepMind Mathematics Dataset into an RL environment based on program synthesis.https://arxiv.org/abs/2107.07373

### [ NASGym](https://github.com/gomerudo/nas-env)

The environment is fully compatible with the OpenAI baselines and exposes a NAS environment following the Neural Structure Code of [BlockQNN: Efficient Block-wise Neural Network Architecture Generation](https://arxiv.org/abs/1808.05584). Under this setting, a Neural Network (i.e. the state for the reinforcement learning agent) is modeled as a list of NSCs, an action is the addition of a layer to the network, and the reward is the accuracy after the early-stop training. The datasets considered so far are the CIFAR-10 dataset (available by default) and the meta-dataset (has to be manually downloaded as specified in [this repository](https://github.com/gomerudo/meta-dataset)).

### [ NLPGym: A toolkit to develop RL agents to solve NLP tasks](https://github.com/rajcscw/nlp-gym)

[NLPGym](https://arxiv.org/pdf/2011.08272v1.pdf) provides interactive environments for standard NLP tasks such as sequence tagging, question answering, and sequence classification. Users can easily customize the tasks with their datasets, observations, features and reward functions.

### [ Obstacle Tower](https://github.com/Unity-Technologies/obstacle-tower-env)

3D procedurally generated tower where you have to climb to the highest level possible

### [ QASGym](https://github.com/qdevpsi3/quantum-arch-search)

This is a list of environments for quantum architecture search following the description in [Quantum Architecture Search via Deep Reinforcement Learning](https://arxiv.org/abs/2104.07715). The agent designs the quantum circuit by taking actions in the environment. Each action corresponds to a gate applied on some wires. The goal is to build a circuit U that generates the target n-qubit quantum state that belongs to the environment and is hidden from the agent. The circuits are built using [Google QuantumAI Cirq](https://quantumai.google/cirq).

### [ mo-gym](https://github.com/LucasAlegre/mo-gym)

Multi-objective RL (MORL) gym environments, where the reward is a NumPy array of different (possibly conflicting) objectives.

### [gym-saturation](https://github.com/inpefess/gym-saturation)

An environment for guiding automated theorem provers based on saturation algorithms (e.g. [Vampire](https://github.com/vprover/vampire)).

### [ShinRL](https://github.com/omron-sinicx/ShinRL/)

ShinRL: A Library for Evaluating RL Algorithms from Theoretical and Practical Perspectives (Deep RL Workshop 2021)

### [racing-rl](https://github.com/luigiberducci/racing-rl/)

reinforcement learning for f1tenth racing

### [ RubiksCubeGym](https://github.com/DoubleGremlin181/RubiksCubeGym)

The RubiksCubeGym package provides environments for twisty puzzles with multiple reward functions to help simulate the methods used by humans.

### [evogym-design-tool](https://github.com/EvolutionGym/evogym-design-tool/)

Design tool for creating Evolution Gym environments.

### [starship-landing-gym](https://github.com/Armandpl/starship-landing-gym/)

A simple Gym environment for propulsive rocket landing

### [RaveForce](https://github.com/chaosprint/RaveForce/)

RaveForce - A Gym style toolkit for music generation experiments.

### [gymnax](https://github.com/RobertTLange/gymnax/)

RL Environments in JAX 🌍
