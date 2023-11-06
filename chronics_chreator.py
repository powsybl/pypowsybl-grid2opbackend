# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See ChronicsCreatorAuthors.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file comes from LightSim2grid, LightSim2grid a implements a c++ backend targeting the Grid2Op platform. It was
# modified to be able to create chronics for pypowsybl backend and takes part of pypowsybl-grid2opbackend.

import json
import shutil
import random
import warnings
import re
import grid2op
import pypowsybl as ppow
import pandapower as pp
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from src.Backend.powsybl_backend import PowsyblBackend
from grid2op.Backend import PandaPowerBackend
from grid2op import make, Parameters
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Chronics import FromNPY
from lightsim2grid import LightSimBackend, TimeSerie, SecurityAnalysis
from tqdm import tqdm
import os
import datetime
import pandas as pd
from src.Backend.network import load as load_ppow_network
from grid2op.Runner import Runner
from grid2op.Agent import DoNothingAgent
from grid2op.Parameters import Parameters

FRAMEWORK = ppow


try:
    from tabulate import tabulate

    TABULATE_AVAIL = True
except ImportError:
    print("The tabulate package is not installed. Some output might not work properly")
    TABULATE_AVAIL = False

VERBOSE = False
MAKE_PLOT = True

case_names = [
    # "case14.json",
    # "case118.json",
    # "case_illinois200.json",
    # "case300.json",
    # "case1354pegase.json",
    "case1888rte.json",
    #"test_1888rte_from_pp.json",
    # "GBnetwork.json",  # 2224 buses
    # "case2848rte.json",
    # "case2869pegase.json",
    # "case3120sp.json",
    # "case6495rte.json",
    # "case6515rte.json",
    # "case9241pegase.json"
]


def make_grid2op_env(Backend, load_p, load_q, gen_p):
    param = Parameters.Parameters()
    param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = make("blank",
                    param=param, test=True,
                    backend=Backend(),
                    chronics_class=FromNPY,
                    data_feeding_kwargs={"load_p": load_p,
                                         "load_q": load_q,
                                         "prod_p": gen_p
                                         },
                    grid_path=case_name,
                    _add_to_name=f"{case_name}",
                    )
    return env


def get_loads_gens(load_p_init, load_q_init, gen_p_init, week, sgen_p_init=None):
    # scale loads

    # use some French time series data for loads
    # see https://github.com/BDonnot/data_generation for where to find this file
    coeffs = {"sources": {
        "country": "France",
        "year": "2012",
        "web": "http://clients.rte-france.com/lang/fr/visiteurs/vie/vie_stats_conso_inst.jsp"
    },
        "month": {
            "jan": 1.21,
            "feb": 1.40,
            "mar": 1.05,
            "apr": 1.01,
            "may": 0.86,
            "jun": 0.84,
            "jul": 0.84,
            "aug": 0.79,
            "sep": 0.85,
            "oct": 0.94,
            "nov": 1.01,
            "dec": 1.20
        },
        "day": {
            "mon": 1.01,
            "tue": 1.05,
            "wed": 1.05,
            "thu": 1.05,
            "fri": 1.03,
            "sat": 0.93,
            "sun": 0.88
        },
        "hour": {
            "00:00": 1.00,
            "01:00": 0.93,
            "02:00": 0.91,
            "03:00": 0.86,
            "04:00": 0.84,
            "05:00": 0.85,
            "06:00": 0.90,
            "07:00": 0.97,
            "08:00": 1.03,
            "09:00": 1.06,
            "10:00": 1.08,
            "11:00": 1.09,
            "12:00": 1.09,
            "13:00": 1.09,
            "14:00": 1.06,
            "15:00": 1.03,
            "16:00": 1.00,
            "17:00": 1.00,
            "18:00": 1.04,
            "19:00": 1.09,
            "20:00": 1.05,
            "21:00": 1.01,
            "22:00": 0.99,
            "23:00": 1.03
        }
    }

    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    week_val = []
    id_month = int(week/4)
    month = months[id_month]
    for day in days:
        # interpolate them at 5 minutes resolution (instead of 1h)
        val = list(coeffs["hour"].values())
        x_final = np.arange(12 * len(val))
        val.append(val[0])
        val = np.array(val) * coeffs["month"][month] * coeffs["day"][day]
        x_interp = 12 * np.arange(len(val))
        coeff_interp = interp1d(x=x_interp, y=val, kind="cubic")
        day_val = coeff_interp(x_final)
        week_val.append(day_val)

    week_val = np.array(week_val)

    # compute the "smooth" loads matrix
    load_p_smooth = week_val.reshape(-1, 1) * load_p_init.reshape(1, -1)
    load_q_smooth = week_val.reshape(-1, 1) * load_q_init.reshape(1, -1)


    # add a bit of noise to it to get the "final" loads matrix
    load_p = load_p_smooth * np.random.lognormal(mean=0., sigma=0.003, size=load_p_smooth.shape)
    load_q = load_q_smooth * np.random.lognormal(mean=0., sigma=0.003, size=load_q_smooth.shape)

    # scale generators accordingly
    gen_p = load_p.sum(axis=1).reshape(-1, 1) / load_p_init.sum() * gen_p_init.reshape(1, -1)
    if sgen_p_init is None or len(sgen_p_init) <= 0 or sgen_p_init.all():
        return load_p, load_q, -gen_p
    else :
        sgen_p = load_p.sum(axis=1).reshape(-1, 1) / load_p_init.sum() * sgen_p_init.reshape(1, -1)
        return load_p, load_q, gen_p, sgen_p

def save_loads_gens(list_columns, list_chronics, save_names):
    """
    Function used to save as csv files the chronics created above with the other get_loads_gens function. The different
    lists should be ordered, so they can correspond adequately (list of loads name combined with list of chronics for load
     combined with the corresponding name to save it)

    :param list_columns: list of names for loads and gens under the format [list_of_loads_name,list_of_loads_name,list_of_gens_name...]
    :type list_columns: :class:`list`

    :param list_chronics: list of chronics for loads and gens under the format [load_p,load_q,prod_p,...] coming from
    get_loads_gens output
    :type list_chronics: :class:`list`

    :param save_names: list of pathnames where to save the load_p/load_q/prod_p... files
    :type save_names: :class: `list`

    :return: ``None``
    """
    try:
        if len(list_columns) != len(list_chronics):
            raise ValueError
        for i in range(len(list_columns)):
            compression_opts = dict(method='bz2')
            df = pd.DataFrame(list_chronics[i], columns=list_columns[i])
            df.to_csv(save_names[i], sep=';', index=False, compression=compression_opts)
    except ValueError:
        print("List does not have the same size, which implies that there are some chronics with unnamed objects")


def prods_charac_creator(back,root_path):
    """
    Create and save the prods_charac.csv file use in chronics.

    :param back: Backend created by Grid2op using pypowsybl
    :type back: :class: PypowsyblBackend

    """
    grid = back._grid
    columns = ['Pmax', 'Pmin', 'name', 'type', 'bus', 'max_ramp_up', 'max_ramp_down', 'min_up_time', 'min_down_time',
               'marginal_cost', 'shut_down_cost', 'start_cost', 'x', 'y', 'V']
    df = pd.DataFrame(columns=columns)
    df['Pmax'] = grid.get_generators(all_attributes=True)['max_p']
    df['Pmin'] = grid.get_generators(all_attributes=True)['min_p']
    df[df['Pmin'] < 0] = 0
    df[df['Pmax'] <= 0] = 10
    df['name'] = grid.get_generators(all_attributes=True).index.values
    df['type'] = 'thermal'
    df['bus'] = [back.map_sub[elem] for elem in grid.get_generators(all_attributes=True)['bus_breaker_bus_id'].values]
    df['max_ramp_up'] = df.apply(ramp_up, axis=1)
    df['max_ramp_down'] = df.apply(ramp_down, axis=1)
    df['min_up_time'] = 4
    df['min_down_time'] = 4
    df['marginal_cost'] = 70
    df['shut_down_cost'] = 1
    df['start_cost'] = 2
    df['V'] = grid.get_generators(all_attributes=True)['target_v']
    df.to_csv(os.path.join(root_path, 'prods_charac.csv'), sep=',', index=False)

def ramp_up(row):
    if row['Pmax'] < 10:
        return row['Pmax']/2
    return 10

def ramp_down(row):
    if row['Pmin'] < 10:
        return row['Pmin']/2
    return 10

def get_env_name_displayed(env_name):
    res = re.sub("^l2rpn_", "", env_name)
    res = re.sub("_small$", "", res)
    res = re.sub("_large$", "", res)
    res = re.sub("\\.json$", "", res)
    return res


def config_file(config_path, thermal_limit=None):
    heading = """from grid2op.Action import PlayableAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFile, GridStateFromFileWithForecasts
from src.Backend.powsybl_backend import PowsyblBackend
    
    """

    if thermal_limit:
        config = """
config = {
    "backend": PowsyblBackend,
    "action_class": PlayableAction,
    "observation_class": None,
    "reward_class": None,  # RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFile,
    "voltagecontroler_class": None,
    "thermal_limits": %s,
    "names_chronics_to_grid": None,
}"""%thermal_limit
    else:
        config = """
config = {
    "backend": PowsyblBackend,
    "action_class": PlayableAction,
    "observation_class": None,
    "reward_class": None,  # RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFile,
    "voltagecontroler_class": None,
    "names_chronics_to_grid": None,
}"""
    to_write = heading + config
    with open((config_path), 'w') as fp:
        fp.write(to_write)

def info_files(date_path, date_formatted):
    time_interval = "00:05"
    time_interval_text_file = open(os.path.join(date_path, "time_interval.info"), "w")
    time_interval_text_file.write(time_interval)
    time_interval_text_file.close()
    start_date_time = str(date_formatted) + " 23:55"
    start_datetime_info_text_file = open(os.path.join(date_path, "start_datetime.info"), "w")
    start_datetime_info_text_file.write(start_date_time)
    start_datetime_info_text_file.close()



def thermal_limit_selection(episode_data):
    """
    This function iterate on all the observations made for all the possible chronics and select for each line a value to
    set for their thermal limit.
    At the end we choose to keep the maximum observed flow value which we are multiplicating by 1.5 and for 10% of the
    rest of the lines we set the thermal limit to the 99 percentile of all observed values.

    Parameters
    ----------
    episode_data : EpisodeData
        The full vector of information coming from a first iteration on all the chronics created

    Returns
    -------

    """
    th_limit_lines = []
    for elem in episode_data:
        for obs in elem[5].observations:
            flow = obs.a_or
            th_limit_lines.append(flow)

    th_limit_lines = np.array(th_limit_lines)
    selected_ind = random.choices(range(len(th_limit_lines[0])), k=int(len(th_limit_lines[0]) * 0.1))
    final_th_limit = []
    for i in range(len(th_limit_lines[0])):
        if i in selected_ind:
            final_th_limit.append(np.percentile(th_limit_lines.transpose()[i], q=99, method='closest_observation'))
        else:
            final_th_limit.append(1.5 * np.max(th_limit_lines.transpose()[i]))
    config_file(config_path, thermal_limit=[x.item() for x in final_th_limit])

if __name__ == "__main__":
    np.random.seed(42)

    case_names_displayed = [get_env_name_displayed(el) for el in case_names]
    g2op_times = []
    g2op_speeds = []
    g2op_sizes = []
    g2op_step_time = []

    ts_times = []
    ts_speeds = []
    ts_sizes = []
    sa_times = []
    sa_speeds = []
    sa_sizes = []

    for case_name in tqdm(case_names):

        if not os.path.exists(case_name):
            import pandapower.networks as pn

            case = getattr(pn, os.path.splitext(case_name)[0])()
            pp.to_json(case, case_name)

        root_path = case_name.split('.')[0]
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        grid_path = os.path.join(root_path, "grid.json")
        shutil.copyfile(case_name, grid_path)

        config_path = os.path.join(root_path, "config.py")
        config_file(config_path)



        # load the case file
        back = PowsyblBackend()

        back.load_grid(grid_path)
        back.runpf(is_dc=True)
        prods_charac_creator(back,root_path)
        coeff_l = 1.0
        # extract reference data
        load_p_init = coeff_l * back._grid.get_loads()["p"].values.astype(dt_float)
        load_q_init = coeff_l * back._grid.get_loads()["q"].values.astype(dt_float)
        gen_p_init = coeff_l * back._grid.get_generators()["p"].values.astype(dt_float)


        res_time = 1.
        res_unit = "s"
        if len(load_p_init) <= 1000:
            # report results in ms if there are less than 1000 loads
            # only affects "verbose" printing
            res_time = 1e3
            res_unit = "ms"

        # simulate the data

        nb_of_week = 2 # choose the number of weeks to create chronics for it is created to have a sequential creation from january to december with 4 weeks per month
        for i in range(nb_of_week):
            load_p, load_q, gen_p = get_loads_gens(load_p_init, load_q_init, gen_p_init, i)
            columns_loads = back._grid.get_loads(all_attributes=True).index.values
            column_gens = back._grid.get_generators(all_attributes=True).index.values
            date_formatted = datetime.datetime(2012, int(i/4)+1, 7*(i%4)+1)
            date_formatted = datetime.date(date_formatted.year, date_formatted.month, date_formatted.day)
            date = os.path.join(root_path, os.path.join("chronics", str(date_formatted))) # format year-nb_of_the_month-nb_of_the_week_in_the_month
            if not os.path.exists(date):
                os.makedirs(date)
            load_p_name = os.path.join(date, 'load_p.csv.bz2')
            load_q_name = os.path.join(date, 'load_q.csv.bz2')
            prod_p_name = os.path.join(date, 'prod_p.csv.bz2')
            save_loads_gens([columns_loads, columns_loads, column_gens], [load_p, load_q, gen_p], [load_p_name, load_q_name, prod_p_name])
            info_files(date, date_formatted)

        p = Parameters()
        p.NO_OVERFLOW_DISCONNECTION = True
        chronics = os.listdir(os.path.join(root_path, "chronics"))

        env = grid2op.make(root_path,
                           backend=PowsyblBackend(detailed_infos_for_cascading_failures=False),
                           param=p)
        runner = Runner(**env.get_params_for_runner(), agentClass=DoNothingAgent)
        episode_data = runner.run(
                        nb_episode=nb_of_week,
                        add_detailed_output=True
                    )

        thermal_limit_selection(episode_data)

