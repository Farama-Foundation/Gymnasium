"""Wrapper that converts the info format for vec envs into the list format."""

from typing import List

import gymnasium as gym


class VectorListInfo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Converts infos of vectorized environments from dict to List[dict].

    This wrapper converts the info format of a
    vector environment from a dictionary to a list of dictionaries.
    This wrapper is intended to be used around vectorized
    environments. If using other wrappers that perform
    operation on info like `RecordEpisodeStatistics` this
    need to be the outermost wrapper.

    i.e. `VectorListInfo(RecordEpisodeStatistics(envs))`

    Example:
        >>> # As dict:
        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }
        >>> # As list:
        >>> infos = [
        ...     {
        ...         "episode": {"r": "<cumulative reward>", "l": "<episode length>", "t": "<elapsed time since beginning of episode>"},
        ...         "final_observation": "<observation>",
        ...         "final_info": {},
        ...     },
        ...     ...,
        ... ]
    """

    def __init__(self, env):
        """This wrapper will convert the info into the list format.

        Args:
            env (Env): The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        try:
            self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            assert False, "This wrapper can only be used in vectorized environments."

    def step(self, action):
        """Steps through the environment, convert dict info to list."""
        observation, reward, terminated, truncated, infos = self.env.step(action)
        list_info = self._convert_info_to_list(infos)

        return observation, reward, terminated, truncated, list_info

    def reset(self, **kwargs):
        """Resets the environment using kwargs."""
        obs, infos = self.env.reset(**kwargs)
        list_info = self._convert_info_to_list(infos)
        return obs, list_info

    def _convert_info_to_list(self, infos: dict) -> List[dict]:
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

    def _process_episode_statistics(self, infos: dict, list_info: list) -> List[dict]:
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
