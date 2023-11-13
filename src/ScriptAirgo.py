# Copyright (c) 2023, Artelys (https://www.artelys.com/)
# @author Rémi Tschupp <remi.tschupp@artelys.com>
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of pypowsybl-grid2opbackend. It is mostly inspired by the development of the several backends from
# Grid2op framework. Some part of codes have been paste/copy.
import os.path

import grid2op

from l2rpn_baselines.PPO_SB3 import train as ppo_train
from l2rpn_baselines.PPO_SB3 import evaluate as ppo_evaluate


from src.Backend.powsybl_backend import PowsyblBackend

DATA_PATH = "data_test\l2rpn_case14_sandbox_Pypowsybl"


def create_train_evaluate():

    env = grid2op.make(
        DATA_PATH,
        backend=PowsyblBackend(detailed_infos_for_cascading_failures=False),
    )

    nm_env_train, nm_env_val = env.train_val_split_random(pct_val=34.)


def train_and_evaluate():

    train_path = DATA_PATH + "_train"
    eval_path = DATA_PATH + "_val"

    if os.path.exists(train_path) and os.path.exists(eval_path):
        pass
    elif not os.path.exists(train_path) and not os.path.exists(eval_path):
        create_train_evaluate()
    else:
        raise FileExistsError("One dataset has been created but not the other one, erase the entire dataset folder and "
                              "start anew")

    train = grid2op.make(train_path, backend=PowsyblBackend(detailed_infos_for_cascading_failures=False))

    agent = ppo_train(train, name="PPO_SB3", save_path="baseline", iterations=10000)

    val = grid2op.make(eval_path, backend=PowsyblBackend(detailed_infos_for_cascading_failures=False))

    g2op_agent, res_val = ppo_evaluate(
        val,
        load_path="baseline/",
        name="PPO_SB3",
        obs_space_kwargs={},
        act_space_kwargs={},
    )

    g2op_agent, res_train = ppo_evaluate(
        train,
        load_path="baseline/",
        name="PPO_SB3",
        obs_space_kwargs={},
        act_space_kwargs={},
        nb_episode=2
    )

    print("On training set results :")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res_train:
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    print("On evaluation set results :")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res_val:
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)


if __name__ == "__main__":

    train_and_evaluate()
