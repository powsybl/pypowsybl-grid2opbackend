import grid2op
from grid2op.Agent import DoNothingAgent
from grid2op.Backend import PandaPowerBackend
from grid2op.PlotGrid import PlotMatplot, PlotPlotly
from tqdm import tqdm

from src.Backend.PowsyblBackend import PowsyblBackend


def run(backend="powsybl", n_iter=1, PlotHelper=PlotMatplot):

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

    backends = ["powsybl", "pandapower"]
    for backend in backends:
        run(backend=backend, n_iter=3)
