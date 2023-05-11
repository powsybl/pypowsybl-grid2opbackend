# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import time

import numpy as np

from grid2op.Episode import EpisodeData
from grid2op.Runner.FakePBar import _FakePbar
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Chronics import ChronicsHandler


def _aux_one_process_parrallel(
    runner,
    episode_this_process,
    process_id,
    path_save=None,
    env_seeds=None,
    agent_seeds=None,
    max_iter=None,
    add_detailed_output=False,
):
    """this is out of the runner, otherwise it does not work on windows / macos"""
    chronics_handler = ChronicsHandler(
        chronicsClass=runner.gridStateclass,
        path=runner.path_chron,
        **runner.gridStateclass_kwargs
    )
    parameters = copy.deepcopy(runner.parameters)
    nb_episode_this_process = len(episode_this_process)
    res = [(None, None, None) for _ in range(nb_episode_this_process)]
    for i, ep_id in enumerate(episode_this_process):
        # `ep_id`: grid2op id of the episode i want to play
        # `i`: my id of the episode played (0, 1, ... episode_this_process)
        env, agent = runner._new_env(
            chronics_handler=chronics_handler, parameters=parameters
        )
        try:
            env_seed = None
            if env_seeds is not None:
                env_seed = env_seeds[i]
            agt_seed = None
            if agent_seeds is not None:
                agt_seed = agent_seeds[i]
            name_chron, cum_reward, nb_time_step, episode_data = _aux_run_one_episode(
                env,
                agent,
                runner.logger,
                ep_id,
                path_save,
                env_seed=env_seed,
                max_iter=max_iter,
                agent_seed=agt_seed,
                detailed_output=add_detailed_output,
            )
            id_chron = chronics_handler.get_id()
            max_ts = chronics_handler.max_timestep()
            if add_detailed_output:
                res[i] = (
                    id_chron,
                    name_chron,
                    float(cum_reward),
                    nb_time_step,
                    max_ts,
                    episode_data,
                )
            else:
                res[i] = (id_chron, name_chron, float(cum_reward), nb_time_step, max_ts)
        finally:
            env.close()
    return res


def _aux_run_one_episode(
    env,
    agent,
    logger,
    indx,
    path_save=None,
    pbar=False,
    env_seed=None,
    agent_seed=None,
    max_iter=None,
    detailed_output=False,
):
    done = False
    time_step = int(0)
    time_act = 0.0
    cum_reward = dt_float(0.0)

    # set the environment to use the proper chronic
    env.set_id(indx)
    # set the seed
    if env_seed is not None:
        env.seed(env_seed)

    # handle max_iter
    if max_iter is not None:
        env.chronics_handler.set_max_iter(max_iter)

    # reset it
    obs = env.reset()

    # seed and reset the agent
    if agent_seed is not None:
        agent.seed(agent_seed)
    agent.reset(obs)

    # compute the size and everything if it needs to be stored
    nb_timestep_max = env.chronics_handler.max_timestep()
    efficient_storing = nb_timestep_max > 0
    nb_timestep_max = max(nb_timestep_max, 0)

    if path_save is None and not detailed_output:
        # i don't store anything on drive, so i don't need to store anything on memory
        nb_timestep_max = 0

    disc_lines_templ = np.full((1, env.backend.n_line), fill_value=False, dtype=dt_bool)

    attack_templ = np.full(
        (1, env._oppSpace.action_space.size()), fill_value=0.0, dtype=dt_float
    )
    if efficient_storing:
        times = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
        rewards = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
        actions = np.full(
            (nb_timestep_max, env.action_space.n), fill_value=np.NaN, dtype=dt_float
        )
        env_actions = np.full(
            (nb_timestep_max, env._helper_action_env.n),
            fill_value=np.NaN,
            dtype=dt_float,
        )
        observations = np.full(
            (nb_timestep_max + 1, env.observation_space.n),
            fill_value=np.NaN,
            dtype=dt_float,
        )
        disc_lines = np.full(
            (nb_timestep_max, env.backend.n_line), fill_value=np.NaN, dtype=dt_bool
        )
        attack = np.full(
            (nb_timestep_max, env._opponent_action_space.n),
            fill_value=0.0,
            dtype=dt_float,
        )
    else:
        times = np.full(0, fill_value=np.NaN, dtype=dt_float)
        rewards = np.full(0, fill_value=np.NaN, dtype=dt_float)
        actions = np.full((0, env.action_space.n), fill_value=np.NaN, dtype=dt_float)
        env_actions = np.full(
            (0, env._helper_action_env.n), fill_value=np.NaN, dtype=dt_float
        )
        observations = np.full(
            (0, env.observation_space.n), fill_value=np.NaN, dtype=dt_float
        )
        disc_lines = np.full((0, env.backend.n_line), fill_value=np.NaN, dtype=dt_bool)
        attack = np.full(
            (0, env._opponent_action_space.n), fill_value=0.0, dtype=dt_float
        )

    need_store_first_act = path_save is not None or detailed_output
    if need_store_first_act:
        # store observation at timestep 0
        if efficient_storing:
            observations[time_step, :] = obs.to_vect()
        else:
            observations = np.concatenate((observations, obs.to_vect().reshape(1, -1)))
    episode = EpisodeData(
        actions=actions,
        env_actions=env_actions,
        observations=observations,
        rewards=rewards,
        disc_lines=disc_lines,
        times=times,
        observation_space=env.observation_space,
        action_space=env.action_space,
        helper_action_env=env._helper_action_env,
        path_save=path_save,
        disc_lines_templ=disc_lines_templ,
        attack_templ=attack_templ,
        attack=attack,
        attack_space=env._opponent_action_space,
        logger=logger,
        name=env.chronics_handler.get_name(),
        force_detail=detailed_output,
        other_rewards=[],
    )
    if need_store_first_act:
        # I need to manually force in the first observation (otherwise it's not computed)
        episode.observations.objects[0] = episode.observations.helper.from_vect(
            observations[time_step, :]
        )
    episode.set_parameters(env)

    beg_ = time.perf_counter()

    reward = float(env.reward_range[0])
    done = False

    next_pbar = [False]
    with _aux_make_progress_bar(pbar, nb_timestep_max, next_pbar) as pbar_:
        while not done:
            beg__ = time.perf_counter()
            act = agent.act(obs, reward, done)
            end__ = time.perf_counter()
            time_act += end__ - beg__

            obs, reward, done, info = env.step(act)  # should load the first time stamp
            cum_reward += reward
            time_step += 1
            pbar_.update(1)
            opp_attack = env._oppSpace.last_attack
            episode.incr_store(
                efficient_storing,
                time_step,
                end__ - beg__,
                float(reward),
                env._env_modification,
                act,
                obs,
                opp_attack,
                info,
            )

        end_ = time.perf_counter()
    episode.set_meta(env, time_step, float(cum_reward), env_seed, agent_seed)

    li_text = [
        "Env: {:.2f}s",
        "\t - apply act {:.2f}s",
        "\t - run pf: {:.2f}s",
        "\t - env update + observation: {:.2f}s",
        "Agent: {:.2f}s",
        "Total time: {:.2f}s",
        "Cumulative reward: {:1f}",
    ]
    msg_ = "\n".join(li_text)
    logger.info(
        msg_.format(
            env._time_apply_act + env._time_powerflow + env._time_extract_obs,
            env._time_apply_act,
            env._time_powerflow,
            env._time_extract_obs,
            time_act,
            end_ - beg_,
            cum_reward,
        )
    )

    episode.set_episode_times(env, time_act, beg_, end_)

    episode.to_disk()
    name_chron = env.chronics_handler.get_name()
    return name_chron, cum_reward, int(time_step), episode


def _aux_make_progress_bar(pbar, total, next_pbar):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Parameters
    ----------
    pbar: ``bool`` or ``type`` or ``object``
        How to display the progress bar, understood as follow:

        - if pbar is ``None`` nothing is done.
        - if pbar is a boolean, tqdm pbar are used, if tqdm package is available and installed on the system
          [if ``true``]. If it's false it's equivalent to pbar being ``None``
        - if pbar is a ``type`` ( a class), it is used to build a progress bar at the highest level (episode) and
          and the lower levels (step during the episode). If it's a type it muyst accept the argument "total"
          and "desc" when being built, and the closing is ensured by this method.
        - if pbar is an object (an instance of a class) it is used to make a progress bar at this highest level
          (episode) but not at lower levels (step during the episode)
    """
    pbar_ = _FakePbar()
    next_pbar[0] = False

    if isinstance(pbar, bool):
        if pbar:
            try:
                from tqdm import tqdm

                pbar_ = tqdm(total=total, desc="episode")
                next_pbar[0] = True
            except (ImportError, ModuleNotFoundError):
                pass
    elif isinstance(pbar, type):
        pbar_ = pbar(total=total, desc="episode")
        next_pbar[0] = pbar
    elif isinstance(pbar, object):
        pbar_ = pbar
    return pbar_
