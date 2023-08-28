from grid2op.Action import TopologyAndDispatchAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAndDispatchAction,
    "observation_class": None,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "thermal_limits": [
        384.900179,
        384.900179,
        380.0,
        380.0,
        157.0,
        380.0,
        380.0,
        1077.7205012,
        461.8802148,
        769.80036,
        269.4301253,
        384.900179,
        760.0,
        380.0,
        760.0,
        384.900179,
        230.9401074,
        170.79945452,
        3402.24266,
        3402.24266,
    ],
    "names_chronics_to_grid": None,
}
