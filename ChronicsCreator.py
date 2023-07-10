# Copyright (c) 2023, Artelys (https://www.artelys.com/)
# @author Rémi Tschupp <remi.tschupp@artelys.com>
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import json
import warnings
import re
import pypowsybl as ppow
import pandapower as pp
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Backend.PowsyblBackend import PowsyblBackend
from grid2op import make, Parameters
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Chronics import FromNPY
from lightsim2grid import LightSimBackend, TimeSerie, SecurityAnalysis
from tqdm import tqdm
import os
import datetime
import pandas as pd
from src.Backend.network import load as load_ppow_network

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


def get_loads_gens(load_p_init, load_q_init, gen_p_init, sgen_p_init=None):
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
    vals = list(coeffs["hour"].values())
    x_final = np.arange(12 * len(vals))

    # interpolate them at 5 minutes resolution (instead of 1h)
    vals.append(vals[0])
    vals = np.array(vals) * coeffs["month"]["oct"] * coeffs["day"]["mon"]
    x_interp = 12 * np.arange(len(vals))
    # start_date_time = datetime.date.fromisocalendar(coeffs.year,)
    coeffs = interp1d(x=x_interp, y=vals, kind="cubic")
    all_vals = coeffs(x_final)

    # compute the "smooth" loads matrix
    load_p_smooth = all_vals.reshape(-1, 1) * load_p_init.reshape(1, -1)
    load_q_smooth = all_vals.reshape(-1, 1) * load_q_init.reshape(1, -1)

    # add a bit of noise to it to get the "final" loads matrix
    load_p = load_p_smooth * np.random.lognormal(mean=0., sigma=0.003, size=load_p_smooth.shape)
    load_q = load_q_smooth * np.random.lognormal(mean=0., sigma=0.003, size=load_q_smooth.shape)

    # scale generators accordingly
    gen_p = load_p.sum(axis=1).reshape(-1, 1) / load_p_init.sum() * gen_p_init.reshape(1, -1)
    if sgen_p_init == None :
        return load_p, load_q, gen_p
    else :
        sgen_p = load_p.sum(axis=1).reshape(-1, 1) / load_p_init.sum() * sgen_p_init.reshape(1, -1)
        return load_p, load_q, gen_p, sgen_p

def save_loads_gens(list_columns,list_chronics,save_names):
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


def prods_charac_creator(back):
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
    df['name'] = grid.get_generators(all_attributes=True).index.values
    df['type'] = 'thermal'
    df['bus'] = [back.map_sub[elem] for elem in grid.get_generators(all_attributes=True)['bus_breaker_bus_id'].values]
    df['max_ramp_up'] = 10
    df['max_ramp_down'] = 10
    df['min_up_time'] = 4
    df['min_down_time'] = 4
    df['marginal_cost'] = 70
    df['shut_down_cost'] = 1
    df['start_cost'] = 2
    df['V'] = grid.get_generators(all_attributes=True)['target_v']
    df.to_csv('prods_charac.csv', sep=',', index=False)


def get_env_name_displayed(env_name):
    res = re.sub("^l2rpn_", "", env_name)
    res = re.sub("_small$", "", res)
    res = re.sub("_large$", "", res)
    res = re.sub("\\.json$", "", res)
    return res



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

        # load the case file
        if FRAMEWORK == ppow:
            back = PowsyblBackend()
            back.load_grid(case_name)
            pandapow_net = pp.from_json(case_name)
            # Handling thermal limits
            with open(r'Thermal_limits.json', 'w') as fp:
                thermal = 1000 * np.concatenate(
                    (
                        pandapow_net.line["max_i_ka"].values,
                        pandapow_net.trafo["sn_mva"].values / (np.sqrt(3) * pandapow_net.trafo["vn_hv_kv"].values)
                    )
                )
                json.dump(list(thermal), fp) # Multiplying by 1000 : kA -> A

            back.runpf(is_dc=True)
            prods_charac_creator(back)
            # extract reference data
            load_p_init = 1.0 * back._grid.get_loads()["p"].values.astype(dt_float)
            load_q_init = 1.0 * back._grid.get_loads()["q"].values.astype(dt_float)
            gen_p_init = 1.0 * back._grid.get_generators()["p"].values.astype(dt_float)

        elif FRAMEWORK == pp:
            case = FRAMEWORK.from_json(case_name)
            FRAMEWORK.runpp(case)  # for slack

            # extract reference data
            load_p_init = 1.0 * case.load["p_mw"].values
            load_q_init = 1.0 * case.load["q_mvar"].values
            gen_p_init = 1.0 * case.gen["p_mw"].values
            sgen_p_init = 1.0 * case.sgen["p_mw"].values

        res_time = 1.
        res_unit = "s"
        if len(load_p_init) <= 1000:
            # report results in ms if there are less than 1000 loads
            # only affects "verbose" printing
            res_time = 1e3
            res_unit = "ms"

        # simulate the data
        if FRAMEWORK == ppow:
            load_p, load_q, gen_p = get_loads_gens(load_p_init, load_q_init, gen_p_init)
            columns_loads = back._grid.get_loads(all_attributes=True).index.values
            column_gens = back._grid.get_generators(all_attributes=True).index.values
            save_loads_gens([columns_loads, columns_loads, column_gens], [load_p, load_q, gen_p], ['load_p.csv.bz2', 'load_q.csv.bz2', 'prod_p.csv.bz2'])

        elif FRAMEWORK == pp:
            load_p, load_q, gen_p, sgen_p = get_loads_gens(load_p_init, load_q_init, gen_p_init, sgen_p_init)

        #save the data

