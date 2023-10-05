import grid2op
from grid2op.Parameters import Parameters
from grid2op.Agent import DoNothingAgent, OneChangeThenNothing
from grid2op.PlotGrid import PlotMatplot, PlotPlotly
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
from tqdm import tqdm
from src.Backend.PowsyblBackend import PowsyblBackend


def run_onechange(acts_dict_=None, nb_of_iterations=5, PlotHelper=PlotMatplot):
    print(f"Run_onechange : " + str(grid2op.__file__))
    env = grid2op.make(
        "data_test\l2rpn_case14_sandbox_Pypowsybl",
        backend=PowsyblBackend(detailed_infos_for_cascading_failures=False),
    )

    env.set_max_iter(nb_of_iterations)
    env.seed(42)
    # plot_helper = PlotHelper(env.observation_space)
    results = dict.fromkeys(acts_dict_.keys())
    for Agent_name, act_as_dict in acts_dict_.items():
        # generate the proper class that will perform the first action (encoded by {}) in acts_dict_
        agent_class = OneChangeThenNothing.gen_next(act_as_dict)
        # start a runner with this agent
        runner = Runner(**env.get_params_for_runner(), agentClass=agent_class)
        # run 2 episode with it
        *_, episode_data = runner.run(nb_episode=1, pbar=True, add_detailed_output=True)[0]
        # fig = plot_helper.plot_obs(
        #    episode_data.observations[1],
        #    line_info="p",
        #    gen_info="p"
        # )

        # if isinstance(plot_helper, PlotPlotly):
        #    fig.write_html(f"Backend {backend} Observation n°{1} - Agent {Agent_name}.html")
        #    fig.update_layout(title=f"Backend {backend} Observation n°{1} - Agent {Agent_name}")
        # else:
        #    fig.suptitle(f"Backend {backend} Observation n°{1} - Agent {Agent_name}")
        #    fig.savefig(f"Grid2Op Backend {backend} Observation n°{1} - Agent {Agent_name}.svg")
        results[Agent_name] = episode_data

    for elem in episode_data.observations:
        print(elem.is_alarm_illegal)
        print(elem.topo_vect)

        # you can do something with it now
    return results


if __name__ == "__main__":

    acts_dict_ = {
        # "Donothing": {},
        # "OneChange_disconnection": {"set_line_status": [(0, -1)]},
        # "OneChange_change_bus": {"set_bus": {"lines_or_id": [(3, 2), (4, 2)], "loads_id": [(0, 2)]}},#, "generators_id": [(0, 2)]}}#{"generators_id": [(0, 2)]}}# {"lines_or_id": [(3, 2), (4, 2)],"loads_id": [(0, 2)]}}#,"generators_id": [(0, 2)]}}
        # "OneRedispatch": {"redispatch":  [(1, -1.3)]}
        # "OneChange_disconnection": {"set_line_status": [(0, -1)]},
        "OneChange_set_bus": {
            "set_bus": {
                "lines_or_id": [(3, 2), (4, 2)],
                "loads_id": [(0, 2)],
                "generators_id": [(0, 2)]
            },
        },
        # "OneChange_change_bus": {
        #     "change_bus": {
        #         "lines_or_id": [3, 4],
        #         "loads_id": [0],
        #         "generators_id": [0]
        #     },
        # }
    }
    results = run_onechange(acts_dict_, nb_of_iterations=15)
