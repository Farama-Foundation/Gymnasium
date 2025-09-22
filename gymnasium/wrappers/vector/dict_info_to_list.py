"""Wrapper that converts the info format for vec envs into the list format."""

from __future__ import annotations

from typing import Any

import numpy as np

from gymnasium.core import ActType, ObsType
from gymnasium.vector.vector_env import ArrayType, VectorEnv, VectorWrapper


__all__ = ["DictInfoToList"]


class DictInfoToList(VectorWrapper):
    """Converts infos of vectorized environments from ``dict`` to ``List[dict]``.

    This wrapper converts the info format of a
    vector environment from a dictionary to a list of dictionaries.
    This wrapper is intended to be used around vectorized
    environments. If using other wrappers that perform
    operation on info like `RecordEpisodeStatistics` this
    need to be the outermost wrapper.

    i.e. ``DictInfoToList(RecordEpisodeStatistics(vector_env))``

    Example:
        >>> import numpy as np
        >>> dict_info = {
        ...      "k": np.array([0., 0., 0.5, 0.3]),
        ...      "_k": np.array([False, False, True, True])
        ...  }
        ...
        >>> list_info = [{}, {}, {"k": 0.5}, {"k": 0.3}]

    Example for vector environments:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> info
        {}
        >>> envs = DictInfoToList(envs)
        >>> obs, info = envs.reset(seed=123)
        >>> info
        [{}, {}, {}]

    Another example for vector environments:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("HalfCheetah-v5", num_envs=2)
        >>> _ = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> _, _, _, _, infos = envs.step(envs.action_space.sample())
        >>> infos
        {'x_position': array([0.03332211, 0.10172355]), '_x_position': array([ True,  True]), 'x_velocity': array([-0.06296527,  0.89345848]), '_x_velocity': array([ True,  True]), 'reward_forward': array([-0.06296527,  0.89345848]), '_reward_forward': array([ True,  True]), 'reward_ctrl': array([-0.24503504, -0.21944423], dtype=float32), '_reward_ctrl': array([ True,  True])}
        >>> envs = DictInfoToList(envs)
        >>> _ = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> _, _, _, _, infos = envs.step(envs.action_space.sample())
        >>> infos  # doctest: +ELLIPSIS
        [{'x_position': np.float64(0.0333221...), 'x_velocity': np.float64(-0.0629652...), 'reward_forward': np.float64(-0.0629652...), 'reward_ctrl': np.float32(-0.2450350...)}, {'x_position': np.float64(0.1017235...), 'x_velocity': np.float64(0.8934584...), 'reward_forward': np.float64(0.8934584...), 'reward_ctrl': np.float32(-0.2194442...)}]

    Change logs:
     * v0.24.0 - Initially added as ``VectorListInfo``
     * v1.0.0 - Renamed to ``DictInfoToList``
    """

    def __init__(self, env: VectorEnv):
        """This wrapper will convert the info into the list format.

        Args:
            env (Env): The environment to apply the wrapper
        """
        super().__init__(env)

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, list[dict[str, Any]]]:
        """Steps through the environment, convert dict info to list."""
        observation, reward, terminated, truncated, infos = self.env.step(actions)
        assert isinstance(infos, dict)
        list_info = self._convert_info_to_list(infos)

        return observation, reward, terminated, truncated, list_info

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, list[dict[str, Any]]]:
        """Resets the environment using kwargs."""
        obs, infos = self.env.reset(seed=seed, options=options)
        assert isinstance(infos, dict)
        list_info = self._convert_info_to_list(infos)

        return obs, list_info

    def _convert_info_to_list(self, vector_infos: dict) -> list[dict[str, Any]]:
        """Convert the dict info to list.

        Convert the dict info of the vectorized environment
        into a list of dictionaries where the i-th dictionary
        has the info of the i-th environment.

        Args:
            vector_infos (dict): info dict coming from the env.

        Returns:
            list_info (list): converted info.
        """
        list_info = [{} for _ in range(self.num_envs)]

        for key, value in vector_infos.items():
            if key.startswith("_"):
                continue

            if isinstance(value, dict):
                value_list_info = self._convert_info_to_list(value)
                for env_num, (env_info, has_info) in enumerate(
                    zip(value_list_info, vector_infos[f"_{key}"])
                ):
                    if has_info:
                        list_info[env_num][key] = env_info
            else:
                assert isinstance(value, np.ndarray)
                for env_num, has_info in enumerate(vector_infos[f"_{key}"]):
                    if has_info:
                        list_info[env_num][key] = value[env_num]

        return list_info
