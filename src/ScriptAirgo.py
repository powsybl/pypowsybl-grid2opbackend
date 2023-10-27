# Copyright (c) 2023, Artelys (https://www.artelys.com/)
# @author Rémi Tschupp <remi.tschupp@artelys.com>
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of pypowsybl-grid2opbackend. It is mostly inspired by the development of the several backends from
# Grid2op framework. Some part of codes have been paste/copy.

import grid2op

from l2rpn_baselines.PPO_SB3 import train as ppo_train
from l2rpn_baselines.PPO_SB3 import evaluate as ppo_evaluate


from src.Backend.powsybl_backend import PowsyblBackend



def train_and_evaluate():
    env = grid2op.make(
        "data_test\l2rpn_case14_sandbox_Pypowsybl",
        backend=PowsyblBackend(detailed_infos_for_cascading_failures=False),
    )

    agent = ppo_train(env, name="PPO_SB3", save_path="baseline", iterations=10000)

    g2op_agent, res = ppo_evaluate(
        env,
        load_path="baseline/",
        name="PPO_SB3",
        nb_episode=3,
        obs_space_kwargs={},
        act_space_kwargs={}
    )
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)


if __name__ == "__main__":

    train_and_evaluate()
