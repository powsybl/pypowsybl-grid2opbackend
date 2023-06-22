import grid2op
from grid2op.Agent import DoNothingAgent, OneChangeThenNothing
from grid2op.Backend import PandaPowerBackend
from grid2op.PlotGrid import PlotMatplot, PlotPlotly
from grid2op.Runner import Runner
from tqdm import tqdm

from src.Backend.PowsyblBackend import PowsyblBackend


def run_onechange(backend="powsybl", acts_dict_=None):

    if backend == "powsybl":
        env = grid2op.make("src\data_test\l2rpn_case14_sandbox_Pypowsybl",
                       backend=PowsyblBackend(detailed_infos_for_cascading_failures=False))
    else:
        env = grid2op.make("l2rpn_case14_sandbox", backend=PandaPowerBackend())

    env.seed(42)
    for act_as_dict in acts_dict_:
        # generate the proper class that will perform the first action (encoded by {}) in acts_dict_
        agent_class = OneChangeThenNothing.gen_next(act_as_dict)

        # start a runner with this agent
        runner = Runner(**env.get_params_for_runner(), agentClass=agent_class)
        # run 2 episode with it
        res_2 = runner.run(nb_episode=1, pbar=True)


def run_donoting(backend="powsybl", n_iter=1, PlotHelper=PlotMatplot):

    if backend == "powsybl":
        env = grid2op.make("src\data_test\l2rpn_case14_sandbox_Pypowsybl",
                       backend=PowsyblBackend(detailed_infos_for_cascading_failures=False))
    else:
        env = grid2op.make("l2rpn_case14_sandbox", backend=PandaPowerBackend())

    env.seed(42)
    plot_helper = PlotHelper(env.observation_space)
    my_agent = DoNothingAgent(env.action_space)

    all_obs = []
    obs = env.reset()
    all_obs.append(obs)
    reward = env.reward_range[0]
    done = False
    nb_step = 0

    with tqdm(total=env.chronics_handler.max_timestep()) as pbar:
        while True:
            action = my_agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            pbar.update(1)
            if done:
                break
            all_obs.append(obs)
            nb_step += 1
            fig = plot_helper.plot_obs(
                obs,
                line_info="p",
                gen_info="p"
            )
            if isinstance(plot_helper, PlotPlotly):
                fig.write_html(f"Backend {backend} Observation n°{nb_step}.html")
                fig.update_layout(title=f"Backend {backend} Observation n°{nb_step}")
            else:
                fig.suptitle(f"Backend {backend} Observation n°{nb_step}")
                fig.savefig(f"Grid2Op Backend {backend} Observation n°{nb_step}.svg")
            if backend == "powsybl":
                env.backend._grid.write_network_area_diagram_svg(f"Pyspowsybl Backend {backend} Observation n°{nb_step}.svg")
            if nb_step == n_iter:
                break
    print("Number of timesteps computed: {}".format(nb_step))
    print("Total maximum number of timesteps possible: {}".format(env.chronics_handler.max_timestep()))


if __name__ == "__main__":
    acts_dict_ = [{}, {"set_line_status": [(0, -1)]}]

    backends = ["pandapower", "pypowsybls"]
    for backend in backends:
        run_onechange(backend, acts_dict_)

