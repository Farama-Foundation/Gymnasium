# Vectorize

## Gymnasium.vector.VectorEnv

### *class* gymnasium.vector.VectorEnv

Base class for vectorized environments to run multiple independent copies of the same environment in parallel.

Vector environments can provide a linear speed-up in the steps taken per second through sampling multiple
sub-environments at the same time. Gymnasium contains two generalised Vector environments: [`AsyncVectorEnv`](vector/async_vector_env.md#gymnasium.vector.AsyncVectorEnv)
and [`SyncVectorEnv`](vector/sync_vector_env.md#gymnasium.vector.SyncVectorEnv) along with several custom vector environment implementations.
For [`reset()`](#gymnasium.vector.VectorEnv.reset) and [`step()`](#gymnasium.vector.VectorEnv.step) batches observations, rewards,  terminations, truncations and
info for each sub-environment, see the example below. For the rewards, terminations, and truncations,
the data is packaged into a NumPy array of shape (num_envs,). For observations (and actions, the batching
process is dependent on the type of observation (and action) space, and generally optimised for neural network
input/outputs. For info, the data is kept as a dictionary such that a key will give the data for all sub-environment.

For creating environments, `make_vec()` is a vector environment equivalent to `make()` for easily creating
vector environments that contains several unique arguments for modifying environment qualities, number of environment,
vectorizer type, vectorizer arguments.

To avoid having to wait for all sub-environments to end before resetting, sub-environments are reset individually
when their episode ends (terminated or truncated is True), with the [`AutoresetMode`](vector/autoreset.md#gymnasium.vector.AutoresetMode) deciding when:
on the next step (`AutoresetMode.NEXT_STEP`, the default for Gymnasium’s implementations), within the same step
(`AutoresetMode.SAME_STEP`) or only when the user calls `reset(options={"reset_mask": mask})`
(`AutoresetMode.DISABLED`). The mode used by a vector environment should be available in `metadata["autoreset_mode"]`.
As the mode changes what [`step()`](#gymnasium.vector.VectorEnv.step) returns at the end of an episode, training code must handle it correctly.
Warning, some vector implementations, wrappers or training algorithms will only support particular autoreset modes.
See [https://gymnasium.farama.org/api/vector/autoreset/](https://gymnasium.farama.org/api/vector/autoreset/) for each mode’s behaviour and example training loops,
and [https://farama.org/Vector-Autoreset-Mode](https://farama.org/Vector-Autoreset-Mode) for background.

#### NOTE
The info parameter of [`reset()`](#gymnasium.vector.VectorEnv.reset) and [`step()`](#gymnasium.vector.VectorEnv.step) was originally implemented before v0.25 as a list
of dictionary for each sub-environment. However, this was modified in v0.25+ to be a dictionary with a NumPy
array for each key. To use the old info style, utilise the `DictInfoToList` wrapper.

### Examples

```pycon
>>> import gymnasium as gym
>>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync", wrappers=(gym.wrappers.TimeAwareObservation,))
>>> envs = gym.wrappers.vector.ClipReward(envs, min_reward=0.2, max_reward=0.8)
>>> envs
<ClipReward, SyncVectorEnv(CartPole-v1, num_envs=3)>
>>> envs.num_envs
3
>>> envs.action_space
MultiDiscrete([2 2 2])
>>> envs.observation_space
Box([[-4.80000019        -inf -0.41887903        -inf  0.        ]
 [-4.80000019        -inf -0.41887903        -inf  0.        ]
 [-4.80000019        -inf -0.41887903        -inf  0.        ]], [[4.80000019e+00            inf 4.18879032e-01            inf
  5.00000000e+02]
 [4.80000019e+00            inf 4.18879032e-01            inf
  5.00000000e+02]
 [4.80000019e+00            inf 4.18879032e-01            inf
  5.00000000e+02]], (3, 5), float64)
>>> observations, infos = envs.reset(seed=123)
>>> observations
array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282,  0.        ],
       [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598,  0.        ],
       [ 0.03517495, -0.000635  , -0.01098382, -0.03203924,  0.        ]])
>>> infos
{}
>>> _ = envs.action_space.seed(123)
>>> actions = envs.action_space.sample()
>>> observations, rewards, terminations, truncations, infos = envs.step(actions)
>>> observations
array([[ 0.01734283,  0.15089367, -0.02859527, -0.33293587,  1.        ],
       [ 0.02909703, -0.16717631,  0.04740972,  0.3319138 ,  1.        ],
       [ 0.03516225, -0.19559774, -0.01162461,  0.25715804,  1.        ]])
>>> rewards
array([0.8, 0.8, 0.8])
>>> terminations
array([False, False, False])
>>> truncations
array([False, False, False])
>>> infos
{}
>>> envs.close()
```

The Vector Environments have the additional attributes for users to understand the implementation

- [`num_envs`](#gymnasium.vector.VectorEnv.num_envs) - The number of sub-environment in the vector environment
- [`observation_space`](#gymnasium.vector.VectorEnv.observation_space) - The batched observation space of the vector environment
- [`single_observation_space`](#gymnasium.vector.VectorEnv.single_observation_space) - The observation space of a single sub-environment
- [`action_space`](#gymnasium.vector.VectorEnv.action_space) - The batched action space of the vector environment
- [`single_action_space`](#gymnasium.vector.VectorEnv.single_action_space) - The action space of a single sub-environment

### Methods

#### VectorEnv.step(actions: ActType) → tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]

Take an action for each parallel environment.

* **Parameters:**
  **actions** – Batch of actions with the [`action_space`](#gymnasium.vector.VectorEnv.action_space) shape.
* **Returns:**
  **Batch of** (*observations, rewards, terminations, truncations, infos*)

#### NOTE
What is returned for a sub-environment whose episode ends depends on the autoreset mode,
`metadata["autoreset_mode"]`, see [https://gymnasium.farama.org/api/vector/autoreset/](https://gymnasium.farama.org/api/vector/autoreset/).

- `AutoresetMode.NEXT_STEP` (default) - The final observation, reward, termination, truncation and info
  are returned as normal. On the next step, the sub-environment’s action is ignored and it is reset,
  returning the reset observation and info with a reward of zero and no termination or truncation.
- `AutoresetMode.SAME_STEP` - The sub-environment is reset within the same step, returning the reset
  observation and info with the final reward, termination and truncation. The final observation and info
  are in `infos["final_obs"]` and `infos["final_info"]`, masked by `infos["_final_obs"]` and
  `infos["_final_info"]`.
- `AutoresetMode.DISABLED` - The final observation, reward, termination, truncation and info are
  returned as normal, and the sub-environment must be reset with `reset(options={"reset_mask": mask})`
  before it is stepped again.

### Example

```pycon
>>> import gymnasium as gym
>>> import numpy as np
>>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
>>> _ = envs.reset(seed=42)
>>> actions = np.array([1, 0, 1], dtype=np.int32)
>>> observations, rewards, terminations, truncations, infos = envs.step(actions)
>>> observations
array([[ 0.02727336,  0.18847767,  0.03625453, -0.26141977],
       [ 0.01431748, -0.24002443, -0.04731862,  0.3110827 ],
       [-0.03822722,  0.1710671 , -0.00848456, -0.2487226 ]],
      dtype=float32)
>>> rewards
array([1., 1., 1.])
>>> terminations
array([False, False, False])
>>> truncations
array([False, False, False])
>>> infos
{}
```

#### VectorEnv.reset(\*, seed: int | None = None, options: dict[str, Any] | None = None) → tuple[ObsType, dict[str, Any]]

Reset all parallel environments and return a batch of initial observations and info.

* **Parameters:**
  * **seed** – The environment reset seed
  * **options** – Additional information to specify how the environments are reset
    (optional, depending on the specific environment)
* **Returns:**
  **A batch of observations and info from the vectorized environment.**

### Example

```pycon
>>> import gymnasium as gym
>>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
>>> observations, infos = envs.reset(seed=42)
>>> observations
array([[ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ],
       [ 0.01522993, -0.04562247, -0.04799704,  0.03392126],
       [-0.03774345, -0.02418869, -0.00942293,  0.0469184 ]],
      dtype=float32)
>>> infos
{}
```

#### VectorEnv.render() → tuple[str | ndarray[Any, Any] | tuple[ndarray[Any, Any], ndarray[Any, Any]], ...] | None

Returns the rendered frames from the parallel environments.

* **Returns:**
  **A tuple of rendered frames from the parallel environments**

#### VectorEnv.close(\*\*kwargs: Any) → None

Close all parallel environments and release resources.

It also closes all the existing image viewers, then calls `close_extras()` and set
[`closed`](#gymnasium.vector.VectorEnv.closed) as `True`.

#### WARNING
This function itself does not close the environments, it should be handled
in `close_extras()`. This is generic for both synchronous and asynchronous
vectorized environments.

#### NOTE
This will be automatically called when garbage collected or program exited.

* **Parameters:**
  **\*\*kwargs** – Keyword arguments passed to `close_extras()`

### Attributes

#### VectorEnv.num_envs *: int*

The number of sub-environments in the vector environment.

#### VectorEnv.action_space *: gym.Space*

The (batched) action space. The input actions of step must be valid elements of action_space.

#### VectorEnv.observation_space *: gym.Space*

The (batched) observation space. The observations returned by reset and step are valid elements of observation_space.

#### VectorEnv.single_action_space *: gym.Space*

The action space of a sub-environment.

#### VectorEnv.single_observation_space *: gym.Space*

The observation space of a sub-environment.

#### VectorEnv.spec *: [EnvSpec](registry.md#gymnasium.envs.registration.EnvSpec) | None* *= None*

The `EnvSpec` of the environment normally set during [`gymnasium.make_vec()`](registry.md#gymnasium.make_vec)

#### VectorEnv.metadata *: dict[str, Any]* *= {}*

The metadata of the environment containing rendering modes, rendering fps, etc

#### VectorEnv.render_mode *: str | None* *= None*

The render mode of the environment which should follow similar specifications to Env.render_mode.

#### VectorEnv.closed *: bool* *= False*

If the vector environment has been closed already.

### Additional Methods

#### *property* VectorEnv.unwrapped *: Self*

Return the base environment.

#### *property* VectorEnv.np_random *: Generator*

Returns the environment’s internal `_np_random` that if not set will initialise with a random seed.

* **Returns:**
  **Instances of \`np.random.Generator\`**

#### *property* VectorEnv.np_random_seed *: int | None*

Returns the environment’s internal `_np_random_seed` that if not set will first initialise with a random int as seed.

If [`np_random_seed`](#gymnasium.vector.VectorEnv.np_random_seed) was set directly instead of through [`reset()`](#gymnasium.vector.VectorEnv.reset) or `set_np_random_through_seed()`,
the seed will take the value -1.

* **Returns:**
  **int** – the seed of the current np_random or -1, if the seed of the rng is unknown

## Autoreset Mode

When a sub-environment’s episode ends, vector environments reset it according to their [`gymnasium.vector.AutoresetMode`](vector/autoreset.md#gymnasium.vector.AutoresetMode): on the next step (`NEXT_STEP`, the default), within the same step (`SAME_STEP`), or only when the user calls `reset(options={"reset_mask": ...})` (`DISABLED`). The mode changes what `step()` returns at the end of an episode and is given by `metadata["autoreset_mode"]`. See [Autoreset Mode](vector/autoreset.md) for each mode’s behaviour and example training loops.

## Making Vector Environments

To create vector environments, gymnasium provides [`gymnasium.make_vec()`](registry.md#gymnasium.make_vec) as an equivalent function to [`gymnasium.make()`](registry.md#gymnasium.make).
