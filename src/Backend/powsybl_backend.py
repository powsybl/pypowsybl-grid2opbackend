# Copyright (c) 2023, Artelys (https://www.artelys.com/)
# See Authors.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of pypowsybl-grid2opbackend. It is mostly inspired by the development of the several backends from
# Grid2op framework. Some part of codes have been paste/copy.

import os  # load the python os default module
import warnings
import numpy as np
import pandas as pd
import pandapower as pdp
import pypowsybl as ppow
import scipy
import copy
import itertools
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Backend.backend import Backend
from grid2op.Exceptions import *
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Action import ActionSpace
from grid2op.Rules import RulesChecker
from src.Backend.network import load as load_ppow_network

BUS_EXTENSION = '_dummy'

class PowsyblBackend(Backend):
    """
    This is a class that is designed to ease the interaction between pypowsybl and grid2op.

    This backend currently does not work with 3 windings transformers, storages and other exotic objects (It could be
    achieved with pypowsybl but the integration is not properly set with grid2op)

    Attributes
    ----------
    p_or: :class:`numpy.array`, dtype:float
        The active power flowing at the origin end of each powerline

    q_or: :class:`numpy.array`, dtype:float
        The reactive power flowing at the origin end of each powerline

    v_or: :class:`numpy.array`, dtype:float
        The voltage magnitude at the origin bus of the powerline

    theta_or: :class:`numpy.array`, dtype:float
        The voltage angle at the origin bus of the powerline

    a_or: :class:`numpy.array`, dtype:float
        The current flowing at the origin end of each powerline

    p_ex: :class:`numpy.array`, dtype:float
        The active power flowing at the extremity end of each powerline

    q_ex: :class:`numpy.array`, dtype:float
        The reactive power flowing at the extremity end of each powerline

    v_ex: :class:`numpy.array`, dtype:float
        The voltage magnitude at the extremity bus of the powerline

    theta_ex: :class:`numpy.array`, dtype:float
        The voltage angle at the extremity bus of the powerline

    a_ex: :class:`numpy.array`, dtype:float
        The current flowing at the extremity end of each powerline

    load_p: :class:`numpy.array`, dtype:float
        The result active load consumption, it is NaN if no loadflow has been computed (MW)

    load_q: :class:`numpy.array`, dtype:float
        The result reactive load consumption, it is NaN if no loadflow has been computed (MVAr)

    load_v: :class:`numpy.array`, dtype:float
        The voltage magnitude at the bus where each load is connected

    load_theta: :class:`numpy.array`, dtype:float
        The voltage angle at the bus where each load is connected

    prod_p: :class:`numpy.array`, dtype:float
        The actual active production of the generator, NaN if no loadflow has been computed (MW)

    prod_q: :class:`numpy.array`, dtype:float
        The actual reactive production of the generator, NaN if no loadflow has been computed (MVAr)

    prod_v: :class:`numpy.array`, dtype:float
        The voltage magnitude at the bus where each generator is connected

    gen_theta: :class:`numpy.array`, dtype:float
        The voltage angle at the bus where each generator is connected

    Examples
    ---------
    The only recommended way to use this class is by passing an instance of a Backend into the "make"
    function of grid2op. Do not attempt to use a backend outside of this specific usage.

    .. code-block:: python

            import grid2op
            from src.Backend.powsybl_backend import PowsyblBackend
            backend = PowsyblBackend()

            env = grid2op.make(backend=backend)
            # and use "env" as any open ai gym environment.

    """

    def __init__(
            self,
            detailed_infos_for_cascading_failures=False,
            dist_slack=False,
            can_be_copied=True,
    ):
        Backend.__init__(
            self,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
            dist_slack=dist_slack,
            can_be_copied=can_be_copied
        )

        self._dist_slack = dist_slack

        self.p_or = None
        self.q_or = None
        self.v_or = None
        self.a_or = None
        self.p_ex = None
        self.q_ex = None
        self.v_ex = None
        self.a_ex = None

        self.load_p = None
        self.load_q = None
        self.load_v = None

        self.storage_p = None
        self.storage_q = None
        self.storage_v = None

        self.prod_p = None
        self.prod_q = None
        self.prod_v = None
        self.line_status = None

        self._nb_bus_before = None

        self.can_output_theta = True  # I support the voltage angle
        self.theta_or = None
        self.theta_ex = None
        self.load_theta = None
        self.gen_theta = None
        self.storage_theta = None

        self._topo_vect = None
        self._init_bus_load = None
        self._init_bus_gen = None
        self._init_bus_lor = None
        self._init_bus_lex = None
        self._get_vector_inj = None
        self._big_topo_to_obj = None
        self._big_topo_to_backend = None
        self.__ppow_backend_initial_grid = None

        # Mapping some fun to apply bus updates
        self._type_to_bus_set = [
            self._apply_load_bus,
            self._apply_gen_bus,
            self._apply_lor_bus,
            self._apply_trafo_hv,
            self._apply_lex_bus,
            self._apply_trafo_lv,
        ]

        self.dim_topo = -1
        self._number_true_line = -1

        self.tol = None
        self.map_sub = {}
        self.__nb_bus_before = None  # number of substation in the powergrid
        self.__nb_powerline = (
            None  # number of powerline (real powerline, not transformer)
        )

    def load_grid(self, path, filename=None):
        """
        Regarding the type of entry file (.json designed for pandapower, .mat for matpower  or .xiidm for pypowsybl)
        we use different kind of loading to be sure that our network is loaded properly.

        Parameters
        ----------
        path : str
            The path to the folder where the grid file is
        filename : str
            The name of the grid file
        Returns
        -------
        """

        if path is None and filename is None:
            raise RuntimeError(
                "You must provide at least one of path or file to load a powergrid."
            )
        if path is None:
            full_path = filename
        elif filename is None:
            full_path = path
        else:
            full_path = os.path.join(path, filename)
        if not os.path.exists(full_path):
            raise RuntimeError('There is no powergrid at "{}"'.format(full_path))

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if full_path.endswith('.json'):
                    pandapow_net = pdp.from_json(full_path)
                    if not pandapow_net.res_bus.shape[
                       0]:  # if there is no info on bus initialize with flat values the matpower network
                       _ = pdp.converter.to_mpc(pandapow_net, full_path.split('.')[0] + '.mat', init='flat')
                    else:
                       _ = pdp.converter.to_mpc(pandapow_net, full_path.split('.')[0] + '.mat')
                    self._grid = load_ppow_network(full_path.split('.')[0] + '.mat',
                                                  {'matpower.import.ignore-base-voltage': 'false'})

                elif full_path.endswith('.mat'):
                    self._grid = load_ppow_network(full_path, {'matpower.import.ignore-base-voltage': 'false'})
                elif full_path.endswith('.xiidm'):
                    self._grid = load_ppow_network(full_path)
                else:
                    raise RuntimeError('This type of file is not handled try a .mat, .xiidm or .json format')
        except Exception as exc_:
            raise BackendError(
                f'Impossible to load the powergrid located at "{full_path}". Please '
                f"check the file exist and that the file represent a valid pypowsybl "
                f"grid. For your information, the error is:\n{exc_}"
            )
        # """
        # Because sometimes we got negative pmin coming from matpower translation
        # """

        # ind = self._grid.get_generators(all_attributes=True).index[self._grid.get_generators(all_attributes=True)['min_p'].values < 0]
        # corresp = [0 for elem in range(len(ind))]
        # self._grid.update_generators(id=ind, min_p=corresp)
        
        lines = self._grid.get_lines(all_attributes=True)
        self.__nb_bus_before = self._grid.get_buses().shape[0]
        self.__nb_powerline = copy.deepcopy(lines[lines["r"] != 0].shape[0])
        self._init_bus_load = self._grid.get_loads(all_attributes=True)["bus_breaker_bus_id"].values
        self._init_bus_gen = self._grid.get_generators(all_attributes=True)["bus_breaker_bus_id"].values
        self._init_bus_lor = lines["bus_breaker_bus1_id"].values
        self._init_bus_lex = lines["bus_breaker_bus2_id"].values

        winding_transfo = self._grid.get_2_windings_transformers(all_attributes=True)
        t_for = winding_transfo["bus_breaker_bus1_id"].values
        t_fex = winding_transfo["bus_breaker_bus2_id"].values
        self._init_bus_lor = np.concatenate((self._init_bus_lor, t_for))
        self._init_bus_lex = np.concatenate((self._init_bus_lex, t_fex))

        # and now initialize the attributes (see list bellow)
        if self._grid.get_3_windings_transformers().shape[0] > 0:
            raise BackendError(f"3 windings transformers are currently not supported. "
                               f"{self._grid.get_3_windings_transformers().shape[0]} found")

        self.n_line = copy.deepcopy(self._grid.get_lines().shape[0]) + \
                      copy.deepcopy(self._grid.get_2_windings_transformers().shape[0])

        df_lines, df_transfo = self._return_real_lines_transfo()
        self.name_line = np.array(
            df_lines["name"].index.to_list() +
            df_transfo["name"].index.to_list()
        )

        self.n_gen = copy.deepcopy(
            self._grid.get_generators().shape[0])  # number of generators in the grid should be read from self._grid
        self.name_gen = np.array(self._grid.get_generators()["name"].index.to_list())

        self.n_load = copy.deepcopy(
            self._grid.get_loads().shape[0])  # number of loads in the grid should be read from self._grid
        self.name_load = np.array(self._grid.get_loads()["name"].index.to_list())
        self.n_sub = copy.deepcopy(
            self._grid.get_buses().shape[0])  # we give as an input the number of buses that seems to be corresponding to substations in Grid2op
        self.topo_per_elem = self._pypowsbyl_bus_name_utility_fct(self._grid)
        self.name_sub = np.array(["sub_{}".format(i) for i in self.topo_per_elem])

        self.n_storage = copy.deepcopy(self._grid.get_batteries().shape[0])

        if self.n_storage == 0:
            self.set_no_storage()  # deactivate storage in grid objects
        else:
            self.name_storage = np.array(self._grid.get_batteries()[
                                             "name"].index.to_list())  # By default in powsybl only one type of storage : batteries

        # the initial thermal limit
        self.thermal_limit_a = None

        list_buses = []
        for elem in self._grid.get_voltage_levels()['name'].index.values:
            list_buses = list_buses+list(self._grid.get_bus_breaker_topology(voltage_level_id=elem).buses.index)

        for i, bus in enumerate(list_buses):
            self.map_sub[bus] = i

        # Doubling the buses for Grid2op necessities
        self._double_buses()

        # Contrarly to Pandapower i do not have to handle issues with slack buses if the files are xiidm and well written
        # I run a powerflow to initialize the paramaters of the grid in the backend
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ppow.loadflow.run_ac(self._grid,
                                 parameters=ppow.loadflow.Parameters(distributed_slack=self._dist_slack))

        # other attributes should be read from self._grid (see table below for a full list of the attributes)
        self._init_private_attrs()

    def reset(self, path=None, grid_filename=None):
        """
        Reload the grid.
        """
        # Assign the content of itself as saved at the end of load_grid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._grid = self.__ppow_backend_initial_grid.deepcopy()
        self._reset_all_nan()
        self._topo_vect[:] = self._get_topo_vect()

    def _init_private_attrs(self):
        """
        Internal function that is used to initialize all grid2op objects from backend, see
        https://grid2op.readthedocs.io/en/latest/space.html#grid2op.Space.GridObjects for more detail.

        We chose to have 2 busbars at each substation to facilitate the integration of our backend with existent tests
        but this remains a personal choice.
        This is done by using the internal function :func: `PowsyblBackend._double_buses`

        I also create some mapping dictionaries to be able to translate bus naming from Grid2op to pypowsybl and vice
        versa
        Returns
        -------
        """
        self.sub_info = np.zeros(self.n_sub, dtype=dt_int)
        self.load_to_subid = np.zeros(self.n_load, dtype=dt_int)
        self.gen_to_subid = np.zeros(self.n_gen, dtype=dt_int)
        self.line_or_to_subid = np.zeros(self.n_line, dtype=dt_int)
        self.line_ex_to_subid = np.zeros(self.n_line, dtype=dt_int)

        self.load_to_sub_pos = np.zeros(self.n_load, dtype=dt_int)
        self.gen_to_sub_pos = np.zeros(self.n_gen, dtype=dt_int)
        self.line_or_to_sub_pos = np.zeros(self.n_line, dtype=dt_int)
        self.line_ex_to_sub_pos = np.zeros(self.n_line, dtype=dt_int)

        if self.n_storage > 0:
            self.storage_to_subid = np.zeros(self.n_storage, dtype=dt_int)
            self.storage_to_sub_pos = np.zeros(self.n_storage, dtype=dt_int)

        pos_already_used = np.zeros(self.n_sub, dtype=dt_int)

        # Allows us to map the id of each substation in Grid2op (an integer) with the name of each corresponding
        add_map = {}
        for bus, bus_position_id in self.map_sub.items():
            add_map[bus + BUS_EXTENSION] = bus_position_id + self.__nb_bus_before
        self.map_sub.update(add_map)

        # For lines & transfos
        df_lines, df_transfo = self._return_real_lines_transfo()
        self.line_or_to_subid = np.array(
            [self.map_sub[i] for i in df_lines["bus_breaker_bus1_id"].values] +
            [self.map_sub[i] for i in df_transfo["bus_breaker_bus1_id"].values])
        self.line_ex_to_subid = np.array(
            [self.map_sub[i] for i in df_lines["bus_breaker_bus2_id"].values] +
            [self.map_sub[i] for i in df_transfo["bus_breaker_bus2_id"].values]
        )

        for i, (line_or_pos_id, line_ex_pos_id) in enumerate(zip(self.line_or_to_subid, self.line_ex_to_subid)):
            self.sub_info[line_or_pos_id] += 1
            self.sub_info[line_ex_pos_id] += 1
            self.line_or_to_sub_pos[i] = pos_already_used[line_or_pos_id]
            self.line_ex_to_sub_pos[i] = pos_already_used[line_ex_pos_id]
            pos_already_used[line_or_pos_id] += 1
            pos_already_used[line_ex_pos_id] += 1

        self._number_true_line = copy.deepcopy(self._grid.get_lines(all_attributes=True)[self._grid.get_lines(all_attributes=True)["r"] != 0].shape[0])

        # For generators
        self.gen_to_subid = np.array(
            [self.map_sub[i] for i in self._grid.get_generators(all_attributes=True)["bus_breaker_bus_id"].to_list()]
        )

        for i, gen_subid in enumerate(self.gen_to_subid):
            self.sub_info[gen_subid] += 1
            self.gen_to_sub_pos[i] = pos_already_used[gen_subid]
            pos_already_used[gen_subid] += 1

        # For loads
        self.load_to_subid = np.array(
            [self.map_sub[i] for i in self._grid.get_loads(all_attributes=True)["bus_breaker_bus_id"].to_list()]
        )

        for i, load_subid in enumerate(self.load_to_subid):
            self.sub_info[load_subid] += 1
            self.load_to_sub_pos[i] = pos_already_used[load_subid]
            pos_already_used[load_subid] += 1

        # For storage
        self.storage_to_subid = np.array(
            [self.map_sub[i] for i in self._grid.get_batteries(all_attributes=True)["bus_breaker_bus_id"].to_list()]
        )

        if self.n_storage > 0:
            for i in range(len(self.storage_to_subid)):
                self.sub_info[self.storage_to_subid[i]] += 1
                self.storage_to_sub_pos[i] = pos_already_used[self.storage_to_subid[i]]
                pos_already_used[self.storage_to_subid[i]] += 1

        self.dim_topo = np.sum(self.sub_info)
        self._compute_pos_big_topo()

        self._get_vector_inj = dict()
        self._get_vector_inj[
            "load_p"
        ] = self._load_grid_load_p_mw
        self._get_vector_inj[
            "load_q"
        ] = self._load_grid_load_q_mvar
        self._get_vector_inj[
            "prod_p"
        ] = self._load_grid_gen_p_mw
        self._get_vector_inj[
            "prod_v"
        ] = self._load_grid_gen_vm

        cpt = 0
        self.thermal_limit_a = np.full(self.n_line, fill_value=1000000, dtype=dt_float)
        operational_limits = self._grid.get_operational_limits()
        current_operational_limits = operational_limits[np.array(self._grid.get_operational_limits()["type"] == "CURRENT")]
        current_and_dur_operational_limits = operational_limits[np.array(self._grid.get_operational_limits()["acceptable_duration"] == -1) & #If this is a permanent limitation, we are not going to take into account other type of limitation
                        np.array(self._grid.get_operational_limits()["type"] == "CURRENT")]
        for elem in self.name_line:
            if elem in current_operational_limits:
                lim_list = []
                #If this is a permanent limitation, we are not going to take into account other type of limitation
                #If this is a limitation on current
                for line_side in current_and_dur_operational_limits.iterrows():
                    lim_list.append(line_side[1]["value"])
                limit = min(lim_list)
                self.thermal_limit_a[cpt] = limit
            cpt += 1

        self.p_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.q_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.v_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.a_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.p_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.q_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.v_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.a_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.line_status = np.full(self.n_line, dtype=dt_bool, fill_value=np.NaN)
        self.load_p = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.load_q = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.load_v = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.prod_p = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.prod_v = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.prod_q = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.storage_p = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self.storage_q = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self.storage_v = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self._nb_bus_before = None

        # shunts data
        self.n_shunt = self._grid.get_shunt_compensators().shape[0]
        self.shunt_to_subid = np.zeros(self.n_shunt, dtype=dt_int) - 1
        self.name_shunt = np.array(self._grid.get_shunt_compensators()["name"].index.to_list())
        self.shunt_to_subid = np.array(
            [self.map_sub[i] for i in self._grid.get_shunt_compensators(all_attributes=True)["bus_breaker_bus_id"].to_list()])
        self._sh_vnkv = self._grid.get_buses()['v_mag'][
            self._grid.get_shunt_compensators()['bus_id'].values].values.astype(
            dt_float
        )
        self.shunts_data_available = True

        # store the topoid -> objid
        self._big_topo_to_obj = [(None, None) for _ in range(self.dim_topo)]
        nm_ = "load"
        for load_id, pos_big_topo in enumerate(self.load_pos_topo_vect):
            self._big_topo_to_obj[pos_big_topo] = (load_id, nm_)
        nm_ = "gen"
        for gen_id, pos_big_topo in enumerate(self.gen_pos_topo_vect):
            self._big_topo_to_obj[pos_big_topo] = (gen_id, nm_)
        nm_ = "lineor"
        for l_id, pos_big_topo in enumerate(self.line_or_pos_topo_vect):
            self._big_topo_to_obj[pos_big_topo] = (l_id, nm_)
        nm_ = "lineex"
        for l_id, pos_big_topo in enumerate(self.line_ex_pos_topo_vect):
            self._big_topo_to_obj[pos_big_topo] = (l_id, nm_)

        # store the topoid -> objid
        self._big_topo_to_backend = [(None, None, None) for _ in range(self.dim_topo)]
        for load_id, pos_big_topo in enumerate(self.load_pos_topo_vect):
            self._big_topo_to_backend[pos_big_topo] = (load_id, load_id, 0)
        for gen_id, pos_big_topo in enumerate(self.gen_pos_topo_vect):
            self._big_topo_to_backend[pos_big_topo] = (gen_id, gen_id, 1)
        for l_id, pos_big_topo in enumerate(self.line_or_pos_topo_vect):
            if l_id < self.__nb_powerline:
                self._big_topo_to_backend[pos_big_topo] = (l_id, l_id, 2)
            else:
                self._big_topo_to_backend[pos_big_topo] = (
                    l_id,
                    l_id - self.__nb_powerline,
                    3,
                )
        for l_id, pos_big_topo in enumerate(self.line_ex_pos_topo_vect):
            if l_id < self.__nb_powerline:
                self._big_topo_to_backend[pos_big_topo] = (l_id, l_id, 4)
            else:
                self._big_topo_to_backend[pos_big_topo] = (
                    l_id,
                    l_id - self.__nb_powerline,
                    5,
                )

        self.theta_or = np.full(self.n_line, fill_value=np.NaN, dtype=dt_float)
        self.theta_ex = np.full(self.n_line, fill_value=np.NaN, dtype=dt_float)
        self.load_theta = np.full(self.n_load, fill_value=np.NaN, dtype=dt_float)
        self.gen_theta = np.full(self.n_gen, fill_value=np.NaN, dtype=dt_float)
        self.storage_theta = np.full(self.n_storage, fill_value=np.NaN, dtype=dt_float)

        self.dim_topo = np.sum(self.sub_info)
        self._compute_pos_big_topo()

        self.line_status[:] = self._get_line_status()
        self._topo_vect = self._get_topo_vect()
        self.tol = 1e-5  # this is NOT the pypowsybl tolerance !!!! this is used to check if a storage unit
        # produce / absorbs anything

        # Create a deep copy of itself in the initial state
        # Store it under super private attribute
        with warnings.catch_warnings():
            # raised on some versions of pandapower / pandas
            warnings.simplefilter("ignore", FutureWarning)
            self.__ppow_backend_initial_grid = self._grid.deepcopy()  # will be initialized in the "assert_grid_correct"

        self.map_sub_invert = {v: k for k, v in self.map_sub.items()}

    def storage_deact_for_backward_comaptibility(self):
        self._init_private_attrs()

    def apply_action(self, backendAction=None):
        """
        Specific implementation of the method to apply an action modifying a powergrid in the pypowsybl format.
        """

        if backendAction is None:
            return
        cls = type(self)

        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backendAction()

        if np.any(prod_p.changed):
            self._grid.update_generators(id=self.name_gen[prod_p.changed], target_p=prod_p.values[prod_p.changed])

        if np.any(prod_v.changed):
            self._grid.update_generators(id=self.name_gen[prod_v.changed], target_v=prod_v.values[prod_v.changed])

        if np.any(load_p.changed):
            self._grid.update_loads(id=self.name_load[load_p.changed], p0=load_p.values[load_p.changed])

        if np.any(load_q.changed):
            self._grid.update_loads(id=self.name_load[load_q.changed], q0=load_q.values[load_q.changed])

        if self.n_storage > 0:
            # active setpoint
            #TODO wrong way to change storage p and topo look up to understand with loads and gens
            raise BackendError("Not ready for production")

            if np.any(storage.changed):
                self._grid.update_batteries(id=self.name_storage[storage.changed])#  target_p=storage.values[storage.changed]) ? not sure of how to use storage.values

            # topology of the storage
            stor_bus = backendAction.get_storages_bus()
            new_bus_id = stor_bus.values[stor_bus.changed]  # id of the busbar 1 or 2 if
            activated = new_bus_id > 0  # mask of storage that have been activated
            new_bus_num = (
                    self.storage_to_subid[stor_bus.changed] + (new_bus_id - 1) * self.n_sub
            )  # bus number
            new_bus_num[~activated] = self.storage_to_subid[stor_bus.changed][
                ~activated
            ]
            self._grid.get_batteries()["connected"].values[stor_bus.changed] = activated
            self._grid.get_batteries(all_attributes=True)["bus_breaker_bus_id"].values[stor_bus.changed] = new_bus_num
            self._topo_vect[self.storage_pos_topo_vect[stor_bus.changed]] = new_bus_num
            self._topo_vect[
                self.storage_pos_topo_vect[stor_bus.changed][~activated]
            ] = -1

        #TODO WIP for the shunts need to have a fix architecture for the buses

        if type(backendAction).shunts_data_available:
            shunt_p, shunt_q, shunt_bus = shunts__

            if np.any(shunt_p.changed):
                self._grid.update_shunt_compensators(id=self.name_shunt[shunt_p.changed],
                                                     p=shunt_p.values[shunt_p.changed])

            if np.any(shunt_q.changed):
                self._grid.update_shunt_compensators(id=self.name_shunt[shunt_q.changed], q=shunt_q.values[shunt_q.changed])

            if np.any(shunt_bus.changed):
                sh_service = shunt_bus.values[shunt_bus.changed] != -1
                self._grid.get_shunt_compensators()["connected"].iloc[shunt_bus.changed] = sh_service
                chg_and_in_service = sh_service & shunt_bus.changed

                for i in range(len(chg_and_in_service)):
                    equipment_name = self.name_shunt[i]
                    if chg_and_in_service[i]:
                        self.move_buses(equipment_name=equipment_name,
                                        bus_or=self._grid.get_shunt_compensators(all_attributes=True).loc[equipment_name]
                                        ["bus_breaker_bus_id"],
                                        bus_dest=int(cls.local_bus_to_global(shunt_bus.values[chg_and_in_service],
                                                                         cls.shunt_to_subid[chg_and_in_service]))
                                        )
        # i made at least a real change, so i implement it in the backend
        for id_el, new_bus in topo__:
            id_el_backend, id_topo, type_obj = self._big_topo_to_backend[id_el]

            if type_obj is not None:
                # storage unit are handled elsewhere
                self._type_to_bus_set[type_obj](new_bus, id_el_backend)

    def _apply_load_bus(self, new_bus, id_el_backend):
        """
         Change load bus.
         If :
             - new_bus_backend<0 : this means that we want to disconnect the load from the bus
             - new_bus_backend>=0 : the load will be connected to the correspondant bus.

        Parameters
        ----------
        new_bus: int
            id of the given new bus in local substation way, could only be 1 or 2
        id_el_backend: int
            id of the specific load in the backend

        Returns
        -------

        """
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self.map_sub[self._init_bus_load[id_el_backend]]
        )
        equipment_name = self.name_load[id_el_backend]
        if new_bus_backend >= 0:
            self.move_buses(equipment_name=equipment_name,
                            bus_or=self._grid.get_loads(all_attributes=True).loc[equipment_name][
                                "bus_breaker_bus_id"],
                            bus_dest=new_bus_backend)
            self._grid.update_loads(id=equipment_name, connected=True)
        else:
            self._grid.update_loads(id=equipment_name, connected=False)

    def _apply_gen_bus(self, new_bus, id_el_backend):
        """
         Change gen bus.
         If :
             - new_bus_backend<0 : this means that we want to disconnect the load from the bus
             - new_bus_backend>=0 : the load will be connected to the correspondant bus.

        Parameters
        ----------
        new_bus: int
            id of the given new bus in local substation way, could only be 1 or 2
        id_el_backend: int
            id of the specific gen in the backend

        Returns
        -------
        """
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self.map_sub[self._init_bus_gen[id_el_backend]]
        )
        equipment_name = self.name_gen[id_el_backend]
        if new_bus_backend >= 0:
            self.move_buses(equipment_name=equipment_name,
                            bus_or=self._grid.get_generators(all_attributes=True).loc[equipment_name]["bus_breaker_bus_id"],
                            bus_dest=new_bus_backend)
            self._grid.update_generators(id=equipment_name, connected=True)
        else:
            self._grid.update_generators(id=equipment_name, connected=False)

    def _apply_lor_bus(self, new_bus, id_el_backend):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self.map_sub[self._init_bus_lor[id_el_backend]]
        )
        self.change_bus_powerline_or(id_el_backend, new_bus_backend)

    def change_bus_powerline_or(self, id_powerline_backend, new_bus_backend):
        """
        Change line origin bus.
        If :
            - new_bus_backend<0 : this means that we want to disconnect the transfo from the bus
            - new_bus_backend>=0 : the transfo will be connected to the correspondant bus.

        Parameters
        ----------
        id_powerline_backend: int
            id of the given line in the backend
        new_bus_backend: int
            id of the new bus for the line to be connected to

        Returns
        -------
        """
        equipment_name = self.name_line[id_powerline_backend]
        if new_bus_backend >= 0:
            self.move_buses(equipment_name=equipment_name,
                            bus_or=self._grid.get_lines(all_attributes=True).loc[equipment_name][
                                "bus_breaker_bus1_id"],
                            bus_dest=new_bus_backend)
            self._grid.update_lines(id=equipment_name, connected1=True)
        else:
            self._grid.update_lines(id=equipment_name, connected1=False)

    def _apply_lex_bus(self, new_bus, id_el_backend):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self.map_sub[self._init_bus_lex[id_el_backend]]
        )
        self.change_bus_powerline_ex(id_el_backend, new_bus_backend)

    def change_bus_powerline_ex(self, id_powerline_backend, new_bus_backend):
        """
        Change line extremity bus.
        If :
            - new_bus_backend<0 : this means that we want to disconnect the transfo from the bus
            - new_bus_backend>=0 : the transfo will be connected to the correspondant bus.

        Parameters
        ----------
        id_powerline_backend: int
            id of the given line in the backend
        new_bus_backend: int
            id of the new bus for the line to be connected to

        Returns
        -------
        """

        equipment_name = self.name_line[id_powerline_backend]
        if new_bus_backend >= 0:
            self.move_buses(equipment_name=equipment_name,
                            bus_or=self._grid.get_lines(all_attributes=True).loc[equipment_name][
                                "bus_breaker_bus2_id"],
                            bus_dest=new_bus_backend)
            self._grid.update_lines(id=equipment_name, connected2=True)
        else:
            self._grid.update_lines(id=equipment_name, connected2=False)

    def _apply_trafo_hv(self, new_bus, id_el_backend):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self.map_sub[self._init_bus_lor[id_el_backend]]
        )
        self.change_bus_trafo_hv(id_el_backend, new_bus_backend)

    def change_bus_trafo_hv(self, id_powerline_backend, new_bus_backend):
        """
        Change high voltage transfo bus. Because of the translation between grid.json (pandapower format) and matpower
        to be used by pypowsybl some transfos are changed to lines and reciprocally, I have to check there type in the
        backend to be sure to use the right functions.
        If :
            - new_bus_backend<0 : this means that we want to disconnect the transfo from the bus
            - new_bus_backend>=0 : the transfo will be connected to the correspondant bus.

        Parameters
        ----------
        id_powerline_backend: int
            id of the given transfo in the backend
        new_bus_backend: int
            id of the new bus for the transfo to be connected to

        Returns
        -------
        """
        # TODO by convention I observed that hv are connected on bus_1 but need to be checked and otherwise do some improvment
        equipment_name = self.name_line[id_powerline_backend]
        type = self._grid.get_identifiables().loc[equipment_name]["type"]
        if new_bus_backend >= 0:
            if type == "LINE":
                self.move_buses(equipment_name=equipment_name,
                                bus_or=self._grid.get_lines(all_attributes=True).loc[equipment_name][
                                    "bus_breaker_bus1_id"],
                                bus_dest=new_bus_backend)
                self._grid.update_lines(id=equipment_name, connected1=True)
            elif type == "TWO_WINDINGS_TRANSFORMER":
                self.move_buses(equipment_name=equipment_name,
                                bus_or=self._grid.get_2_windings_transformers(all_attributes=True).loc[equipment_name][
                                    "bus_breaker_bus1_id"],
                                bus_dest=new_bus_backend)
                self._grid.update_2_windings_transformers(id=equipment_name, connected1=True)
            else:
                raise BackendError(f"The elements named {equipment_name} is not a transfo")
        else:
            if type == "LINE":
                self._grid.update_lines(id=equipment_name, connected1=False)
            elif type == "TWO_WINDINGS_TRANSFORMER":
                self._grid.update_2_windings_transformers(id=equipment_name, connected1=False)
            else:
                raise BackendError(f"The elements named {equipment_name} is not a transfo")

    def _apply_trafo_lv(self, new_bus, id_el_backend):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self.map_sub[self._init_bus_lex[id_el_backend]]
        )
        self.change_bus_trafo_lv(id_el_backend, new_bus_backend)

    def change_bus_trafo_lv(self, id_powerline_backend, new_bus_backend):
        """
        Change low voltage transfo bus. Because of the translation between grid.json (pandapower format) and matpower
        to be used by pypowsybl some transfos are changed to lines and reciprocally, I have to check there type in the
        backend to be sure to use the right functions.
        If :
            - new_bus_backend<0 : this means that we want to disconnect the transfo from the bus
            - new_bus_backend>=0 : the transfo will be connected to the correspondant bus.

        Parameters
        ----------
        id_powerline_backend: int
            id of the given transfo in the backend
        new_bus_backend: int
            id of the new bus for the transfo to be connected to

        Returns
        -------
        """
        # TODO by convention I observed that lv are connected on bus_2 but need to be checked and otherwise do some improvment

        equipment_name = self.name_line[id_powerline_backend]
        type = self._grid.get_identifiables().loc[equipment_name]["type"]
        if new_bus_backend >= 0:
            if type == "LINE":
                self.move_buses(equipment_name=equipment_name,
                                bus_or=self._grid.get_lines(all_attributes=True).loc[equipment_name][
                                    "bus_breaker_bus2_id"],
                                bus_dest=new_bus_backend)
                self._grid.update_lines(id=equipment_name, connected2=True)
            elif type == "TWO_WINDINGS_TRANSFORMER":
                self.move_buses(equipment_name=equipment_name,
                                bus_or=self._grid.get_2_windings_transformers(all_attributes=True).loc[equipment_name]["bus_breaker_bus2_id"],
                                bus_dest=new_bus_backend)
                self._grid.update_2_windings_transformers(id=equipment_name, connected2=True)
            else:
                raise BackendError(f"The elements named {equipment_name} is not a transfo")
        else:
            if type == "LINE":
                self._grid.update_lines(id=equipment_name, connected2=False)
            elif type == "TWO_WINDINGS_TRANSFORMER":
                self._grid.update_2_windings_transformers(id=equipment_name, connected2=False)
            else:
                raise BackendError(f"The elements named {equipment_name} is not a transfo")

    def runpf(self, is_dc=False):
        """

        Run a power flow on the underlying _grid.
        """

        nb_bus = self.get_nb_active_bus()

        try:
            with warnings.catch_warnings():
                # remove the warning if _grid non connex. And it that case load flow as not converged
                warnings.filterwarnings(
                    "ignore", category=scipy.sparse.linalg.MatrixRankWarning
                )
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)

                if nb_bus == self._nb_bus_before:
                    self._pf_init = ppow._pypowsybl.VoltageInitMode.PREVIOUS_VALUES
                else:
                    self._pf_init = ppow._pypowsybl.VoltageInitMode.DC_VALUES

                if np.any(~self._grid.get_loads()["connected"]):
                    # TODO see if there is a better way here -> do not handle this here, but rather in Backend._next_grid_state
                    raise BackendError("Disconnected load: for now grid2op cannot handle properly"
                                       " disconnected load. If you want to disconnect one, say it"
                                       " consumes 0. instead. Please check loads: "
                                       f"{np.where(~self._grid.get_loads()['connected'])[0]}"
                                       )
                if np.any(~self._grid.get_generators()["connected"]):
                    # TODO see if there is a better way here -> do not handle this here, but rather in Backend._next_grid_state
                    raise BackendError("Disconnected gen: for now grid2op cannot handle properly"
                                       " disconnected generators. If you want to disconnect one, say it"
                                       " produces 0. instead. Please check generators: "
                                       f"{np.where(~self._grid.get_generators()['connected'])[0]}"
                                       )

                if is_dc:
                    res = ppow.loadflow.run_dc(self._grid,
                                               parameters=ppow.loadflow.Parameters(distributed_slack=self._dist_slack))
                else:
                    res = ppow.loadflow.run_ac(self._grid,
                                               parameters=ppow.loadflow.Parameters(distributed_slack=self._dist_slack,
                                                                                   voltage_init_mode=self._pf_init
                                                                                   ))

                # TODO handle the cases where things are disconnected
                (
                    self.prod_p[:],
                    self.prod_q[:],
                    self.prod_v[:],
                    self.gen_theta[:],
                ) = self._gens_info()
                (
                    self.load_p[:],
                    self.load_q[:],
                    self.load_v[:],
                    self.load_theta[:],
                ) = self._loads_info()
                
                branches = self._grid.get_branches()
                branches_bus1 = branches['bus1_id'] 
                branches_bus2 = branches['bus2_id']

                self.p_or[:] = self._aux_get_line_info("p1")
                self.q_or[:] = self._aux_get_line_info("q1")
                self.v_or[:] = self._aux_get_voltage_info(branches_bus1)
                self.a_or[:] = self._aux_get_line_info("i1")
                self.theta_or[:] = self._aux_get_theta_info(branches_bus1)
                self.a_or[~np.isfinite(self.a_or)] = 0.0
                self.v_or[~np.isfinite(self.v_or)] = 0.0

                self.p_ex[:] = self._aux_get_line_info("p2")
                self.q_ex[:] = self._aux_get_line_info("q2")
                self.v_ex[:] = self._aux_get_voltage_info(branches_bus2)
                self.a_ex[:] = self._aux_get_line_info("i2")
                self.theta_ex[:] = self._aux_get_theta_info(branches_bus2)
                self.a_ex[~np.isfinite(self.a_ex)] = 0.0
                self.v_ex[~np.isfinite(self.v_ex)] = 0.0


                # handle storage units
                # note that we have to look ourselves for disconnected storage
                (
                    self.storage_p[:],
                    self.storage_q[:],
                    self.storage_v[:],
                    self.storage_theta[:],
                ) = self._storages_info()
                deact_storage = ~np.isfinite(self.storage_v)
                if np.any(np.abs(self.storage_p[deact_storage]) > self.tol):
                    raise BackendError(
                        "Isolated storage set to absorb / produce something"
                    )
                self.storage_p[deact_storage] = 0.0
                self.storage_q[deact_storage] = 0.0
                self.storage_v[deact_storage] = 0.0
                self._grid.get_batteries()["connected"].values[deact_storage] = False

                self.line_status[:] = self._get_line_status()
                self._topo_vect[:] = self._get_topo_vect()


                if res[0].status == ppow._pypowsybl.LoadFlowComponentStatus.FAILED\
                        or res[0].status == ppow._pypowsybl.LoadFlowComponentStatus.MAX_ITERATION_REACHED:
                    return False, None
                else:
                    return True, None

        except BackendError as exc_:
            # of the powerflow has not converged, results are Nan
            self._reset_all_nan()
            msg = exc_.__str__()
            return False, DivergingPowerFlow(f'powerflow diverged with error :"{msg}"')

    def get_line_status(self):
        """

        As all the functions related to powerline, pypowsybl split them into multiple objects to access with separated
        getters (some for transformers, some for lines etc.). We make sure to get them all here.
        """
        return self.line_status

    def _get_line_status(self):
        lines = self._grid.get_lines(all_attributes=True)
        connected_1_lines = lines[lines["r"] != 0]['connected1']
        connected_2_lines = lines[lines["r"] != 0]['connected2']
        line_connected = connected_1_lines.values & connected_2_lines.values

        windings_transfo = self._grid.get_2_windings_transformers()
        connected_1_2_transfo = pd.concat([windings_transfo['connected1'], lines[lines["r"] == 0]['connected1']])
        connected_2_2_transfo = pd.concat([windings_transfo['connected2'], lines[lines["r"] == 0]['connected2']])
        transfo_2_connected = connected_1_2_transfo.values & connected_2_2_transfo.values

        # We do not take into account 3 windings transfos for the moment

        return np.concatenate(
            (
                line_connected,
                transfo_2_connected,
            )
        ).astype(dt_bool)
        
    def _aux_get_voltage_info(self, elements):
        buses = self._grid.get_buses()['v_mag']
        v_list = [buses[elt] if elt != '' else 0 for elt in elements]
        return v_list

    def _aux_get_theta_info(self, elements):
        buses = self._grid.get_buses()['v_angle']
        v_list = [buses[elt] if elt != '' else 0 for elt in elements]
        return v_list

    def get_topo_vect(self):
        return self._topo_vect

    def _get_topo_vect(self):
        res = np.full(self.dim_topo, fill_value=np.nan, dtype=dt_int)

        line_status = self.get_line_status()

        i = 0
        for row in self._grid.get_lines(all_attributes=True)[self._grid.get_lines(all_attributes=True)["r"] != 0][["bus_breaker_bus1_id", "bus_breaker_bus2_id"]].values:

            if line_status[i]:
                bus_or_id = self.map_sub[row[0]]
                bus_ex_id = self.map_sub[row[1]]
                res[self.line_or_pos_topo_vect[i]] = (
                    1 if bus_or_id == self.line_or_to_subid[i] else 2
                )
                res[self.line_ex_pos_topo_vect[i]] = (
                    1 if bus_ex_id == self.line_ex_to_subid[i] else 2
                )
            else:
                res[self.line_or_pos_topo_vect[i]] = -1
                res[self.line_ex_pos_topo_vect[i]] = -1
            i += 1

        nb = self._number_true_line

        # For 2 windings transfo
        i = 0
        """
        Because there is a problem of transcription between pandapower and pypowsybl using matpower I have to give all 
        the elements considered as transformers 
        """
        for row in np.concatenate((self._grid.get_2_windings_transformers(all_attributes=True)[["bus_breaker_bus1_id", "bus_breaker_bus2_id"]].values, self._grid.get_lines(all_attributes=True)[self._grid.get_lines(all_attributes=True)["r"] == 0][["bus_breaker_bus1_id", "bus_breaker_bus2_id"]].values), axis=0):
            j = i + nb
            if line_status[j]:
                bus_or_id = self.map_sub[row[0]]
                bus_ex_id = self.map_sub[row[1]]
                res[self.line_or_pos_topo_vect[j]] = (
                    1 if bus_or_id == self.line_or_to_subid[j] else 2
                )
                res[self.line_ex_pos_topo_vect[j]] = (
                    1 if bus_ex_id == self.line_ex_to_subid[j] else 2
                )
            else:
                res[self.line_or_pos_topo_vect[j]] = -1
                res[self.line_ex_pos_topo_vect[j]] = -1
            i += 1

        # We do not handle three windings transfos

        i = 0
        for bus_id in self._grid.get_generators(all_attributes=True)["bus_breaker_bus_id"].values:
            res[self.gen_pos_topo_vect[i]] = 1 if self.map_sub[bus_id] == self.gen_to_subid[i] else 2
            i += 1

        i = 0
        for bus_id in self._grid.get_loads(all_attributes=True)["bus_breaker_bus_id"].values:
            res[self.load_pos_topo_vect[i]] = (
                1 if self.map_sub[bus_id] == self.load_to_subid[i] else 2
            )
            i += 1

        if self.n_storage:
            # storage can be deactivated by the environment for backward compatibility
            i = 0
            for bus_id in self._grid.get_batteries(all_attributes=True)["bus_breaker_bus_id"].values:
                status = self._grid.get_batteries()["connected"].values[i]
                if status:
                    res[self.storage_pos_topo_vect[i]] = (
                        1 if self.map_sub[bus_id] == self.storage_to_subid[i] else 2
                    )
                else:
                    res[self.storage_pos_topo_vect[i]] = -1
                i += 1

        return res

    def shunt_info(self):
        shunt_p = self._grid.get_shunt_compensators()["p"].values.astype(dt_float)
        shunt_q = self._grid.get_shunt_compensators()["q"].values.astype(dt_float)
        shunt_v = self._grid.get_buses()['v_mag'][self._grid.get_shunt_compensators()['bus_id']].values.astype(dt_float)
        shunt_bus = self.global_bus_to_local(
            np.array(
                [self.map_sub[elem] for elem in
                 self._grid.get_shunt_compensators(all_attributes=True)["bus_breaker_bus_id"].values]
            ),
            self.shunt_to_subid)
        shunt_v[~self._grid.get_shunt_compensators()["connected"].values] = -1.0
        shunt_bus[~self._grid.get_shunt_compensators()["connected"].values] = -1
        return shunt_p, shunt_q, shunt_v, shunt_bus

    def storages_info(self):
        return (
            copy.deepcopy(self.storage_p),
            copy.deepcopy(self.storage_q),
            copy.deepcopy(self.storage_v),
        )

    def _storages_info(self):
        if self.n_storage:
            # this is because we support "backward comaptibility" feature. So the storage can be
            # deactivated from the Environment...
            p_storage = self._grid.get_batteries()["p"].values.astype(dt_float)
            q_storage = self._grid.get_batteries()["q"].values.astype(dt_float)
            v_storage = self._aux_get_voltage_info(self._grid.get_batteries()['bus_id'])
            theta_storage = self._aux_get_theta_info(self._grid.get_batteries()['bus_id'])
        else:
            p_storage = np.zeros(shape=0, dtype=dt_float)
            q_storage = np.zeros(shape=0, dtype=dt_float)
            v_storage = np.zeros(shape=0, dtype=dt_float)
            theta_storage = np.zeros(shape=0, dtype=dt_float)
        return p_storage, q_storage, v_storage, theta_storage

    def generators_info(self):
        return (
            copy.deepcopy(self.prod_p),
            copy.deepcopy(self.prod_q),
            copy.deepcopy(self.prod_v),
        )

    def _gens_info(self):
        prod_p = - self._grid.get_generators()["p"].values.astype(dt_float)
        prod_q = - self._grid.get_generators()["q"].values.astype(dt_float)
        prod_v = self._aux_get_voltage_info(self._grid.get_generators()['bus_id'])
        prod_theta = self._aux_get_theta_info(self._grid.get_generators()['bus_id'])

        return copy.deepcopy(prod_p), copy.deepcopy(prod_q), copy.deepcopy(prod_v), copy.deepcopy(prod_theta)

    def loads_info(self):
        return (
            copy.deepcopy(self.load_p),
            copy.deepcopy(self.load_q),
            copy.deepcopy(self.load_v)
        )

    def _loads_info(self):
        load_p = self._grid.get_loads()["p"].values.astype(dt_float)
        load_q = self._grid.get_loads()["q"].values.astype(dt_float)
        load_v = self._aux_get_voltage_info(self._grid.get_loads()['bus_id'])
        load_theta = self._aux_get_theta_info(self._grid.get_loads()['bus_id'])
        return load_p, load_q, load_v, load_theta

    def lines_or_info(self):
        return (
            copy.deepcopy(self.p_or),
            copy.deepcopy(self.q_or),
            copy.deepcopy(self.v_or),
            copy.deepcopy(self.a_or)
        )

    def lines_ex_info(self):
        return (
            copy.deepcopy(self.p_ex),
            copy.deepcopy(self.q_ex),
            copy.deepcopy(self.v_ex),
            copy.deepcopy(self.a_ex)
        )

    def get_theta(self):
        return (
            copy.deepcopy(self.theta_or),
            copy.deepcopy(self.theta_ex),
            copy.deepcopy(self.load_theta),
            copy.deepcopy(self.gen_theta),
            copy.deepcopy(self.storage_theta),
        )

    def assert_grid_correct(self):
        """
            This is done as it should be by the Environment
        """
        super().assert_grid_correct()

    def _aux_get_line_info(self, colname):
        lines = self._grid.get_lines(all_attributes=True)
        winding_transfo = self._grid.get_2_windings_transformers(all_attributes=True)
        res = np.concatenate(
            (
                lines[lines["r"] != 0][colname].values,
                winding_transfo[colname].values,
                lines[lines["r"] == 0][colname].values,
            )
        )
        return res

    def _return_real_lines_transfo(self):
        """
        Allows to retrieve the same order as in pandapower with json files, because some transformers (the one with low
        voltage and not any tap change possible) are considered as lines by pypowsybl
        """
        lines = self._grid.get_lines(all_attributes=True)
        return lines[lines["r"] != 0], \
            pd.concat([self._grid.get_2_windings_transformers(all_attributes=True), lines[lines["r"] == 0]], join='inner')

    def sub_from_bus_id(self, bus_id):
        # TODO check that the function is doing what we want
        if bus_id >= self._number_true_line:
            return bus_id - self._number_true_line
        return bus_id

    def get_nb_active_bus(self):
        """
        Compute the amount of buses "in service" eg with at least a powerline connected to it.

        Returns
        -------
        res: :class:`int`
            The total number of active buses.
        """
        return len(self._grid.get_buses())

    def _reset_all_nan(self):
        self.p_or[:] = np.NaN
        self.q_or[:] = np.NaN
        self.v_or[:] = np.NaN
        self.a_or[:] = np.NaN
        self.p_ex[:] = np.NaN
        self.q_ex[:] = np.NaN
        self.v_ex[:] = np.NaN
        self.a_ex[:] = np.NaN
        self.prod_p[:] = np.NaN
        self.prod_q[:] = np.NaN
        self.prod_v[:] = np.NaN
        self.load_p[:] = np.NaN
        self.load_q[:] = np.NaN
        self.load_v[:] = np.NaN
        self.storage_p[:] = np.NaN
        self.storage_q[:] = np.NaN
        self.storage_v[:] = np.NaN
        self._nb_bus_before = None

        self.theta_or[:] = np.NaN
        self.theta_ex[:] = np.NaN
        self.load_theta[:] = np.NaN
        self.gen_theta[:] = np.NaN
        self.storage_theta[:] = np.NaN

    @staticmethod
    def _load_grid_load_p_mw(grid):
        return grid.get_loads()["p"]

    @staticmethod
    def _load_grid_load_q_mvar(grid):
        return grid.get_loads()["q"]

    @staticmethod
    def _load_grid_gen_p_mw(grid):
        return grid.get_generators()["p"]

    @staticmethod
    def _load_grid_gen_vm(grid):
        return grid.get_buses()['v_mag'][grid.get_generators()['bus_id']]

    def _pypowsbyl_bus_name_utility_fct(self, grid):
        """
        Function use to access all the name of the buses in bus_breaker_topology way.

        Parameters
        ----------
        grid
            Pypowsybl grid object

        Returns
        -------
        res: :class:`List`
            List of all the buses in format bus_breaker_topology
        """
        vol_levels = [list(grid.get_bus_breaker_topology(voltage_level_id=elem).buses.index) for elem in grid.get_voltage_levels().index]
        L = list(itertools.chain.from_iterable(vol_levels))
        #L = [bus_id for elem in grid.get_voltage_levels().index for bus_id in grid.get_bus_breaker_topology(voltage_level_id=elem).buses.index]
        #L = []
        #for elem in grid.get_voltage_levels().index:
        #    for bus_id in grid.get_bus_breaker_topology(voltage_level_id=elem).buses.index:
        #        L.append(bus_id)
        return L

    def _double_buses(self):
        """
        Double the buses in the pypowybl backend to ensure that Grid2op framework is working as desired. Modifies
        directly the underlying grid.
        Returns
        -------
        """

        df = self._grid.get_buses()
        #L = self._pypowsbyl_bus_name_utility_fct(self._grid)
        #L = []
        #for elem in grid.get_voltage_levels().index:
        #    for bus_id in self._grid.get_bus_breaker_topology(voltage_level_id=elem).buses.index:
        #        L.append(bus_id)
        L_voltage_id = df['voltage_level_id'].to_list()
        for i in range(len(self.topo_per_elem)):
            self._grid.create_buses(id=self.topo_per_elem[i] + BUS_EXTENSION, voltage_level_id=L_voltage_id[i], name=df['name'][i])

    def move_buses(self, equipment_name, bus_or, bus_dest):
        """
        Function use to call the move_connectable method for every kind of powsybl object. I use self.map_sub_invert to
        get the name of the destination bus in pypowsybl norm (not Grid2op)

        Parameters
        ----------
        equipment_name: str
            Name of the equipment in pypowsybl backend context
        bus_or: str
            The name of the origin bus in pypowsybl backend context
        bus_dest: str
            The name of the destination bus in Grid2op context
        Returns
        -------

        """

        real_bus_dest = self.map_sub_invert[bus_dest]
        if real_bus_dest != bus_or:
            ppow.network.move_connectable(network=self._grid,
                                          equipment_id=equipment_name,
                                          bus_origin_id=bus_or,
                                          bus_destination_id=real_bus_dest)

    def copy(self):
        """
        Performs a deep copy of the power :attr:`_grid`.
        """
        res = type(self)(**self._my_kwargs)

        # copy from base class (backend)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            res._grid = self._grid.deepcopy()
        res.thermal_limit_a = copy.deepcopy(self.thermal_limit_a)
        res._sh_vnkv = copy.deepcopy(self._sh_vnkv)
        res.can_output_theta = self.can_output_theta
        res._is_loaded = self._is_loaded

        # copy all attributes from myself
        res.map_sub = copy.deepcopy(self.map_sub)
        res.map_sub_invert = copy.deepcopy(self.map_sub_invert)

        res.p_or = copy.deepcopy(self.p_or)
        res.q_or = copy.deepcopy(self.q_or)
        res.v_or = copy.deepcopy(self.v_or)
        res.a_or = copy.deepcopy(self.a_or)
        res.p_ex = copy.deepcopy(self.p_ex)
        res.q_ex = copy.deepcopy(self.q_ex)
        res.v_ex = copy.deepcopy(self.v_ex)
        res.a_ex = copy.deepcopy(self.a_ex)

        res.load_p = copy.deepcopy(self.load_p)
        res.load_q = copy.deepcopy(self.load_q)
        res.load_v = copy.deepcopy(self.load_v)

        res.storage_p = copy.deepcopy(self.storage_p)
        res.storage_q = copy.deepcopy(self.storage_q)
        res.storage_v = copy.deepcopy(self.storage_v)

        res.prod_p = copy.deepcopy(self.prod_p)
        res.prod_q = copy.deepcopy(self.prod_q)
        res.prod_v = copy.deepcopy(self.prod_v)
        res.line_status = copy.deepcopy(self.line_status)

        res._pf_init = self._pf_init
        res._nb_bus_before = self._nb_bus_before

        res.thermal_limit_a = copy.deepcopy(self.thermal_limit_a)

        res._number_true_line = self._number_true_line
        res.dim_topo = self.dim_topo
        res._topo_vect = copy.deepcopy(self._topo_vect)

        # function to rstore some information
        res.__nb_bus_before = (
            self.__nb_bus_before
        )  # number of substation in the powergrid
        res.__nb_powerline = (
            self.__nb_powerline
        )  # number of powerline (real powerline, not transformer)
        res._init_bus_load = copy.deepcopy(self._init_bus_load)
        res._init_bus_gen = copy.deepcopy(self._init_bus_gen)
        res._init_bus_lor = copy.deepcopy(self._init_bus_lor)
        res._init_bus_lex = copy.deepcopy(self._init_bus_lex)
        res._get_vector_inj = copy.deepcopy(self._get_vector_inj)
        res._big_topo_to_obj = copy.deepcopy(self._big_topo_to_obj)
        res._big_topo_to_backend = copy.deepcopy(self._big_topo_to_backend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            res.__ppow_backend_initial_grid = self.__ppow_backend_initial_grid.deepcopy()

        res.tol = (
            self.tol
        )  # this is NOT the pypowsybl tolerance !!!! this is used to check if a storage unit
        # produce / absorbs anything

        # TODO storage doc (in grid2op rst) of the backend
        res.theta_or = copy.deepcopy(self.theta_or)
        res.theta_ex = copy.deepcopy(self.theta_ex)
        res.load_theta = copy.deepcopy(self.load_theta)
        res.gen_theta = copy.deepcopy(self.gen_theta)
        res.storage_theta = copy.deepcopy(self.storage_theta)

        return res

    def close(self):
        """

        Called when the :class:`grid2op;Environment` has terminated, this function only reset the grid to a state
        where it has not been loaded.
        """
        del self._grid
        self._grid = None
        del self.__ppow_backend_initial_grid
        self.__ppow_backend_initial_grid = None

    def _disconnect_line(self, id_):
        """
        Function only use in unittest
        """
        game_rules = RulesChecker()
        self._topo_vect[self.line_or_pos_topo_vect[id_]] = -1
        self._topo_vect[self.line_ex_pos_topo_vect[id_]] = -1
        self.line_status[id_] = False
        action_env_class = ActionSpace.init_grid(self)
        action_env = action_env_class(
            gridobj=self, legal_action=game_rules.legal_action
        )
        action = action_env({"change_line_status": id_})
        bk_class = _BackendAction.init_grid(self)
        bk_action = bk_class()
        bk_action += action
        self.apply_action(backendAction=bk_action)

    def _reconnect_line(self, id_):
        """
        Function only use in unittest
        """
        self._topo_vect[self.line_or_pos_topo_vect[id_]] = 1
        self._topo_vect[self.line_ex_pos_topo_vect[id_]] = 1
        self.line_status[id_] = True
        game_rules = RulesChecker()
        action_env_class = ActionSpace.init_grid(self)
        action_env = action_env_class(
            gridobj=self, legal_action=game_rules.legal_action
        )
        action = action_env({"change_line_status": id_})
        bk_class = _BackendAction.init_grid(self)
        bk_action = bk_class()
        bk_action += action
        self.apply_action(backendAction=bk_action)
