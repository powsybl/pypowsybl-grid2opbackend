import grid2op
from grid2op.Agent import DoNothingAgent
from grid2op.PlotGrid import PlotMatplot
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.Backend.PowsyblBackend import PowsyblBackend

if __name__ == "__main__":

    env = grid2op.make("src\data_test\l2rpn_case14_sandbox_Pypowsybl", backend=PowsyblBackend(detailed_infos_for_cascading_failures=False))
    env.seed(42)
    plot_helper = PlotMatplot(env.observation_space)
    my_agent = DoNothingAgent(env.action_space)

    all_obs = []
    obs = env.reset()
    all_obs.append(obs)
    reward = env.reward_range[0]
    done = False
    nb_step = 0
    try:
        with tqdm(total=env.chronics_handler.max_timestep()) as pbar:
            while True:
                action = my_agent.act(obs, reward, done)
                obs, reward, done, _ = env.step(action)
                pbar.update(1)
                if done:
                    break
                all_obs.append(obs)
                nb_step += 1
    except Exception:
        pass
    print("Number of timesteps computed: {}".format(nb_step))
    print("Total maximum number of timesteps possible: {}".format(env.chronics_handler.max_timestep()))
    last_obs = all_obs[-1]
    for i, obs in enumerate(all_obs):
        fig = plot_helper.plot_obs(obs)
        fig.suptitle(f"Observation n°{i}")
        plt.show()
