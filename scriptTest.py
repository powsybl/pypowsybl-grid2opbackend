import grid2op
from grid2op.PlotGrid import PlotMatplot
from grid2op.Agent import DoNothingAgent
from grid2op.Episode import EpisodeData
import numpy as np
import os
import shutil
from grid2op.gym_compat import GymEnv
from gym import Env
from gym.utils.env_checker import check_env
import tqdm
from grid2op.Runner import Runner

import sys
sys.path.append('..\src')
from Backend.PowsyblBackend import PowsyblBackend

env = grid2op.make("D:\Projets\AIRGo\OfficialRepo\pypowsybl-grid2opbackend\src\data_test\l2rpn_case14_sandbox_Pypowsybl",backend = PowsyblBackend(detailed_infos_for_cascading_failures=False))
max_iter = 5  # we limit the number of iterations to reduce computation time. Put -1 if you don't want to limit it
env.seed(42)
obs = env.reset()