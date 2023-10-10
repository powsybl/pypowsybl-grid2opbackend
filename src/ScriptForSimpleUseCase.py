# Copyright (c) 2023, Artelys (https://www.artelys.com/)
# @author Rémi Tschupp <remi.tschupp@artelys.com>
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of pypowsybl-grid2opbackend. It is mostly inspired by the development of the several backends from
# Grid2op framework. Some part of codes have been paste/copy.

import grid2op
from grid2op.Agent import OneChangeThenNothing
from grid2op.PlotGrid import PlotMatplot, PlotPlotly
from grid2op.Runner import Runner
from src.Backend.PowsyblBackend import PowsyblBackend

def run_onechange(acts_dict_=None, nb_of_iterations=5, PlotHelper=PlotMatplot):
    print(f"Run_onechange : " + str(grid2op.__file__))
    env = grid2op.make(
        "data_test\l2rpn_case14_sandbox_Pypowsybl",
        backend=PowsyblBackend(detailed_infos_for_cascading_failures=False),
    )

    env.set_max_iter(nb_of_iterations)
    env.seed(42)
    plot_helper = PlotHelper(env.observation_space)
    for Agent_name, act_as_dict in acts_dict_.items():
        # generate the proper class that will perform the first action (encoded by {}) in acts_dict_
        agent_class = OneChangeThenNothing.gen_next(act_as_dict)
        # start a runner with this agent
        runner = Runner(**env.get_params_for_runner(), agentClass=agent_class)
        # run 1 episode with it
        *_, episode_data = runner.run(nb_episode=1, pbar=True, add_detailed_output=True)[0]
        fig = plot_helper.plot_obs(
           episode_data.observations[1],
           line_info="p",
           gen_info="p"
        )

        if isinstance(plot_helper, PlotPlotly):
           fig.write_html(f"Observation n°{1} - Agent {Agent_name}.html")
           fig.update_layout(title=f"Observation n°{1} - Agent {Agent_name}")
        else:
           fig.suptitle(f"Observation n°{1} - Agent {Agent_name}")
           fig.savefig(f"Observation n°{1} - Agent {Agent_name}.svg")

    # you can do something with it now
    for elem in episode_data.observations:
        print(elem.is_alarm_illegal)
        print(elem.topo_vect)

if __name__ == "__main__":

    acts_dict_ = {
        # "Donothing": {},
        # "OneChange_disconnection": {"set_line_status": [(0, -1)]},
        # "OneRedispatch": {"redispatch":  [(1, -1.3)]}
        # "OneChange_disconnection": {"set_line_status": [(0, -1)]},
        "OneChange_set_bus": {
            "set_bus": {
                "lines_or_id": [(3, 2), (4, 2)],
                "loads_id": [(0, 2)],
                "generators_id": [(1, 2)]
            },
        # },
        # "OneChange_change_bus": {
        #     "change_bus": {
        #         "lines_or_id": [3, 4],
        #         "loads_id": [0],
        #         "generators_id": [1]
        #     },
        }
    }
    run_onechange(acts_dict_, nb_of_iterations=10)
