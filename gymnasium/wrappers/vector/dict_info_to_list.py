"""Wrapper that converts the info format for vec envs into the list format."""
from __future__ import annotations

from typing import Any

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
        >>> import gymnasium as gym
        >>> from gymnasium.spaces import Dict, Box
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3)
        >>> obs, info = envs.reset(seed=123)
        >>> info
        {}
        >>> envs = DictInfoToList(envs)
        >>> obs, info = envs.reset(seed=123)
        >>> info
        [{}, {}, {}]

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
        list_info = self._convert_info_to_list(infos)

        return obs, list_info

    def _convert_info_to_list(self, infos: dict) -> list[dict[str, Any]]:
        """Convert the dict info to list.

        Convert the dict info of the vectorized environment
        into a list of dictionaries where the i-th dictionary
        has the info of the i-th environment.

        Args:
            infos (dict): info dict coming from the env.

        Returns:
            list_info (list): converted info.

        """
        list_info = [{} for _ in range(self.num_envs)]
        list_info = self._process_episode_statistics(infos, list_info)
        for k in infos:
            if k.startswith("_"):
                continue
            for i, has_info in enumerate(infos[f"_{k}"]):
                if has_info:
                    list_info[i][k] = infos[k][i]
        return list_info

    # todo - I think this function should be more general for any information
    def _process_episode_statistics(self, infos: dict, list_info: list) -> list[dict]:
        """Process episode statistics.

        `RecordEpisodeStatistics` wrapper add extra
        information to the info. This information are in
        the form of a dict of dict. This method process these
        information and add them to the info.
        `RecordEpisodeStatistics` info contains the keys
        "r", "l", "t" which represents "cumulative reward",
        "episode length", "elapsed time since instantiation of wrapper".

        Args:
            infos (dict): infos coming from `RecordEpisodeStatistics`.
            list_info (list): info of the current vectorized environment.

        Returns:
            list_info (list): updated info.

        """
        episode_statistics = infos.pop("episode", False)
        if not episode_statistics:
            return list_info

        episode_statistics_mask = infos.pop("_episode")
        for i, has_info in enumerate(episode_statistics_mask):
            if has_info:
                list_info[i]["episode"] = {}
                list_info[i]["episode"]["r"] = episode_statistics["r"][i]
                list_info[i]["episode"]["l"] = episode_statistics["l"][i]
                list_info[i]["episode"]["t"] = episode_statistics["t"][i]

        return list_info
