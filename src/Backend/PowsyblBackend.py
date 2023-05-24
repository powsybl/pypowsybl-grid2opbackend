# Copyright (c) 2023, Artelys (https://www.artelys.com/)
# @author Rémi Tschupp <remi.tschupp@artelys.com>
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os  # load the python os default module
import sys  # laod the python sys default module
import copy
import warnings
import numpy as np
import pandas as pd

import pandapower as pdp
import pypowsybl as ppow
import scipy
import copy

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Backend.Backend import Backend
from grid2op.Action import BaseAction
from grid2op.Exceptions import *

try:
    import numba

    numba_ = True
except (ImportError, ModuleNotFoundError):
    numba_ = False
    warnings.warn(
        "Numba cannot be loaded. You will gain possibly massive speed if installing it by "
        "\n\t{} -m pip install numba\n".format(sys.executable)
    )


class PowsyblBackend(Backend):

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

        self.prod_pu_to_kv = None
        self.load_pu_to_kv = None
        self.lines_or_pu_to_kv = None
        self.lines_ex_pu_to_kv = None
        self.storage_pu_to_kv = None

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

        self.theta_or = None
        self.theta_ex = None
        self.load_theta = None
        self.gen_theta = None
        self.storage_theta = None

        self._iref_slack = None

        self._topo_vect = None

        self._get_vector_inj = None

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

        self.__nb_bus_before = None # number of substation in the powergrid




    def load_grid(self, path, filename=None):

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

        """
        TODO : check for file extension
        Following : https://pypowsybl.readthedocs.io/en/stable/user_guide/network.html
        pow.network.get_import_formats()
        ['CGMES', 'MATPOWER', 'IEEE-CDF', 'PSS/E', 'UCTE', 'XIIDM', 'POWER-FACTORY']
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if full_path.endswith('.json'):
                pandapow_net = pdp.from_json(full_path)
                if not pandapow_net.res_bus.shape[0]: #if there is no info on bus initialize with flat values the matpower network
                    _ = pdp.converter.to_mpc(pandapow_net, full_path.split('.')[0]+'.mat', init='flat')
                else:
                    _ = pdp.converter.to_mpc(pandapow_net, full_path.split('.')[0]+'.mat')
                self._grid = ppow.network.load(full_path.split('.')[0]+'.mat', {'matpower.import.ignore-base-voltage': 'false'})
            elif full_path.endswith('.mat'):
                self._grid = ppow.network.load(full_path, {'matpower.import.ignore-base-voltage': 'false'})
            elif full_path.endswith('.xiidm'):
                self._grid = ppow.network.load(full_path)
            else:
                raise RuntimeError('This type of file is not handled try a .mat, .xiidm or .json format')

        #TODO to see if we really have to double the bus for grid2op vision

        # """
        # We want here to change the network
        # """
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore")
        #     try:
        #         ppow.loadflow.run_ac(
        #             self._grid,
        #             distributed_slack=self._dist_slack
        #         )
        #     except ppow.powerflow.LoadflowNotConverged:
        #         ppow.loadflow.run_dc(
        #             self._grid,
        #             distributed_slack=self._dist_slack,
        #         )


        # and now initialize the attributes (see list bellow)
        self.n_line = copy.deepcopy(self._grid.get_lines().shape[0]) + \
                    copy.deepcopy(self._grid.get_2_windings_transformers().shape[0]) + \
                    copy.deepcopy(self._grid.get_3_windings_transformers().shape[0]) # we add the transfos because they are not modeled in grid2op
        self.n_gen = copy.deepcopy(self._grid.get_generators().shape[0])  # number of generators in the grid should be read from self._grid
        self.n_load = copy.deepcopy(self._grid.get_loads().shape[0])  # number of loads in the grid should be read from self._grid
        self.n_sub = copy.deepcopy(self._grid.get_buses().shape[0])  # we give as an input the number of buses that seems to be corresponding to substations in Grid2op
        self.n_storage = copy.deepcopy(self._grid.get_batteries().shape[0])


        # TODO protection against empty columns
        self.name_load = np.array(self._grid.get_loads()["name"].index.to_list())
        self.name_gen = np.array(self._grid.get_generators()["name"].index.to_list())
        self.name_line = np.array(self._grid.get_lines()["name"].index.to_list()+
                                  self._grid.get_2_windings_transformers()["name"].index.to_list()+
                                  self._grid.get_3_windings_transformers()["name"].index.to_list())
        self.name_sub = np.array(["sub_{}".format(i) for i in self._pypowsbyl_bus_name_utility_fct(self._grid)])

        if self.n_storage == 0:
            self.set_no_storage() #deactivate storage in grid objects
        else:
            self.name_storage = np.array(self._grid.get_batteries()[
                                             "name"].index.to_list())  # By default in powsybl only one type of storage : batteries

        # print(self._grid.get_operational_limits(all_attributes=True))

        # print(self._grid.get_2_windings_transformers())

        # the initial thermal limit
        self.thermal_limit_a = None

        # other attributes should be read from self._grid (see table below for a full list of the attributes)
        self._init_private_attrs()

        # print(self._grid.get_lines()["name"].index.to_list())
        self.__nb_bus_before = self._grid.get_buses().shape[0]
        self.__nb_powerline = self._grid.get_lines().shape[0]

        # and finish the initialization with a call to this function
        self._compute_pos_big_topo()

        self._double_buses()

    def _init_private_attrs(self):

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

        #TODO handle storage and other non classical objects

        pos_already_used = np.zeros(self.n_sub, dtype=dt_int)

        # Allows us to map the id of each substation in Grid2op (an integer) with the name of each corresponding bus in
        # Pypowsybl
        self.map_sub = {bus: i for i, (bus, row) in enumerate(self._grid.get_buses().iterrows())}


        # For lines
        self.line_or_to_subid = np.array([self.map_sub[i] for i in self._grid.get_lines()["bus1_id"].to_list()] +
                                         [self.map_sub[i] for i in
                                          self._grid.get_2_windings_transformers()["bus1_id"].to_list()] +
                                         [self.map_sub[i] for i in
                                          self._grid.get_3_windings_transformers()["bus1_id"].to_list()])
        for i in range(len(self.line_or_to_subid)):
            self.sub_info[self.line_or_to_subid[i]] += 1
            self.line_or_to_sub_pos[i] = pos_already_used[self.line_or_to_subid[i]]
            pos_already_used[self.line_or_to_subid[i]] += 1
            # pos_already_used[sub_or_id] += 1

        self.line_ex_to_subid = np.array([self.map_sub[i] for i in self._grid.get_lines()["bus2_id"].to_list()] +
                                         [self.map_sub[i] for i in
                                          self._grid.get_2_windings_transformers()["bus2_id"].to_list()] +
                                         [self.map_sub[i] for i in
                                          self._grid.get_3_windings_transformers()["bus2_id"].to_list()])

        for i in range(len(self.line_ex_to_subid)):
            self.sub_info[self.line_ex_to_subid[i]] += 1
            self.line_ex_to_sub_pos[i] = pos_already_used[self.line_ex_to_subid[i]]
            pos_already_used[self.line_ex_to_subid[i]] += 1

        self._number_true_line = copy.deepcopy(self._grid.get_lines().shape[0])

        # For generators
        self.gen_to_subid = np.array([self.map_sub[i] for i in self._grid.get_generators()["bus_id"].to_list()])

        for i in range(len(self.gen_to_subid)):
            self.sub_info[self.gen_to_subid[i]] += 1
            self.gen_to_sub_pos[i] = pos_already_used[self.gen_to_subid[i]]
            pos_already_used[self.gen_to_subid[i]] += 1
        # for i, (_, row) in enumerate(self._grid.get_generators().iterrows()):

        # For loads
        self.load_to_subid = np.array([self.map_sub[i] for i in self._grid.get_loads()["bus_id"].to_list()])

        for i in range(len(self.load_to_subid)):
            self.sub_info[self.load_to_subid[i]] += 1
            self.load_to_sub_pos[i] = pos_already_used[self.load_to_subid[i]]
            pos_already_used[self.load_to_subid[i]] += 1

        # For storage
        self.storage_to_subid = np.array([self.map_sub[i] for i in self._grid.get_batteries()["bus_id"].to_list()])

        if self.n_storage > 0:
            for i in range(len(self.storage_to_subid)):
                self.sub_info[self.storage_to_subid[i]] += 1
                self.storage_to_sub_pos[i] = pos_already_used[self.storage_to_subid[i]]
                pos_already_used[self.storage_to_subid[i]] += 1


        self._get_vector_inj = {}
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

        #TODO check the other lines of code in PandaPowerBackend

        self.theta_or = np.full(self.n_line, fill_value=np.NaN, dtype=dt_float)
        self.theta_ex = np.full(self.n_line, fill_value=np.NaN, dtype=dt_float)
        self.load_theta = np.full(self.n_load, fill_value=np.NaN, dtype=dt_float)
        self.gen_theta = np.full(self.n_gen, fill_value=np.NaN, dtype=dt_float)
        self.storage_theta = np.full(self.n_storage, fill_value=np.NaN, dtype=dt_float)

        # for i, (_, row) in enumerate(self._grid.get_loads().iterrows()):
        self.dim_topo = np.sum(self.sub_info)
        self._compute_pos_big_topo()


        #TODO find thermal limitation in matpower import because this is only a hack

        self.thermal_limit_a = np.array([1000000]*len(self.line_or_to_subid))
        self.thermal_limit_a = self.thermal_limit_a.astype(dt_float)

        self.line_status[:] = self._get_line_status()
        self._topo_vect = self._get_topo_vect()
        self.tol = 1e-5  # this is NOT the pandapower tolerance !!!! this is used to check if a storage unit
        # produce / absorbs anything

    def storage_deact_for_backward_comaptibility(self):
        self._init_private_attrs()

    def apply_action(self, backendAction=None):
        if backendAction is None:
            return
        cls = type(self)

        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backendAction()

        # print((prod_p.values, prod_v.values, load_p.values, load_q.values))
        # print((prod_p.changed, prod_v.changed, load_p.changed, load_q.changed))

        #TODO Normally we don't have to handle this for the backend because inactive buses will not appear in
        # get_buses() fct for pypowsybl

        # handle bus status
        # bus_is = self._grid.get_buses()
        # for i, (bus1_status, bus2_status) in enumerate(active_bus):
        #     bus_is[i] = bus1_status  # no iloc for bus, don't ask me why please :-/
        #     bus_is[i + self.__nb_bus_before] = bus2_status

        tmp_prod_p = self._get_vector_inj["prod_p"](self._grid)
        if np.any(prod_p.changed):
            tmp_prod_p.iloc[prod_p.changed] = prod_p.values[prod_p.changed]

        tmp_prod_v = self._get_vector_inj["prod_v"](self._grid)
        if np.any(prod_v.changed):
            tmp_prod_v.iloc[prod_v.changed] = (
                    prod_v.values[prod_v.changed] # / self.prod_pu_to_kv[prod_v.changed]
            )

        if self._id_bus_added is not None and prod_v.changed[self._id_bus_added]:
            # handling of the slack bus, where "2" generators are present.
            self._grid["ext_grid"]["vm_pu"] = 1.0 * tmp_prod_v[self._id_bus_added]

        tmp_load_p = self._get_vector_inj["load_p"](self._grid)
        if np.any(load_p.changed):
            tmp_load_p.iloc[load_p.changed] = load_p.values[load_p.changed]

        tmp_load_q = self._get_vector_inj["load_q"](self._grid)
        if np.any(load_q.changed):
            tmp_load_q.iloc[load_q.changed] = load_q.values[load_q.changed]

        if self.n_storage > 0:
            # active setpoint
            tmp_stor_p = self._grid.storage["p_mw"]
            if np.any(storage.changed):
                tmp_stor_p.iloc[storage.changed] = storage.values[storage.changed]

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
            self._grid.storage["in_service"].values[stor_bus.changed] = activated
            self._grid.storage["bus"].values[stor_bus.changed] = new_bus_num
            self._topo_vect[self.storage_pos_topo_vect[stor_bus.changed]] = new_bus_num
            self._topo_vect[
                self.storage_pos_topo_vect[stor_bus.changed][~activated]
            ] = -1

        #TODO handle shunts

        # if type(backendAction).shunts_data_available:
        #     shunt_p, shunt_q, shunt_bus = shunts__
        #
        #     if np.any(shunt_p.changed):
        #         self._grid.shunt["p_mw"].iloc[shunt_p.changed] = shunt_p.values[
        #             shunt_p.changed
        #         ]
        #     if np.any(shunt_q.changed):
        #         self._grid.shunt["q_mvar"].iloc[shunt_q.changed] = shunt_q.values[
        #             shunt_q.changed
        #         ]
        #     if np.any(shunt_bus.changed):
        #         sh_service = shunt_bus.values[shunt_bus.changed] != -1
        #         self._grid.shunt["in_service"].iloc[shunt_bus.changed] = sh_service
        #         chg_and_in_service = sh_service & shunt_bus.changed
        #         self._grid.shunt["bus"].loc[chg_and_in_service] = cls.local_bus_to_global(
        #             shunt_bus.values[chg_and_in_service],
        #             cls.shunt_to_subid[chg_and_in_service])

        # i made at least a real change, so i implement it in the backend
        for id_el, new_bus in topo__:
            id_el_backend, id_topo, type_obj = self._big_topo_to_backend[id_el]

            if type_obj is not None:
                # storage unit are handled elsewhere
                self._type_to_bus_set[type_obj](new_bus, id_el_backend, id_topo)

    def _apply_load_bus(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_load[id_el_backend]
        )
        if new_bus_backend >= 0:
            self._grid.load["bus"].iat[id_el_backend] = new_bus_backend
            self._grid.load["in_service"].iat[id_el_backend] = True
        else:
            self._grid.load["in_service"].iat[id_el_backend] = False
            self._grid.load["bus"].iat[id_el_backend] = -1

    def _apply_gen_bus(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_gen[id_el_backend]
        )
        if new_bus_backend >= 0:
            self._grid.gen["bus"].iat[id_el_backend] = new_bus_backend
            self._grid.gen["in_service"].iat[id_el_backend] = True
            # remember in this case slack bus is actually 2 generators for pandapower !
            if (
                id_el_backend == (self._grid.gen.shape[0] - 1)
                and self._iref_slack is not None
            ):
                self._grid.ext_grid["bus"].iat[0] = new_bus_backend
        else:
            self._grid.gen["in_service"].iat[id_el_backend] = False
            self._grid.gen["bus"].iat[id_el_backend] = -1
            # in this case the slack bus cannot be disconnected

    def _apply_lor_bus(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_lor[id_el_backend]
        )
        self.change_bus_powerline_or(id_el_backend, new_bus_backend)

    def _apply_lex_bus(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_lex[id_el_backend]
        )
        self.change_bus_powerline_ex(id_el_backend, new_bus_backend)

    def _apply_trafo_hv(self, new_bus, id_el_backend, id_topo):
        new_bus_backend = type(self).local_bus_to_global_int(
            new_bus, self._init_bus_lor[id_el_backend]
        )
        self.change_bus_trafo_hv(id_topo, new_bus_backend)

    def runpf(self, is_dc=False):
        nb_bus = self.get_nb_active_bus()

        try:
            with warnings.catch_warnings():
                # remove the warning if _grid non connex. And it that case load flow as not converged
                warnings.filterwarnings(
                    "ignore", category=scipy.sparse.linalg.MatrixRankWarning
                )
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)

                # TODO see if there is a way of initialazing the calculus of the solver

                # if self._nb_bus_before is None:
                #     self._pf_init = "dc"
                # elif nb_bus == self._nb_bus_before:
                #     self._pf_init = "results"
                # else:
                #     self._pf_init = "auto"

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
                    res = ppow.loadflow.run_dc(self._grid, parameters=ppow.loadflow.Parameters(distributed_slack=self._dist_slack))
                else:
                    res = ppow.loadflow.run_ac(self._grid, parameters=ppow.loadflow.Parameters(distributed_slack=self._dist_slack))
                    # print(self._dist_slack)
                    # print(res)

                # TODO check how to handle

                # # stores the computation time
                # if "_ppc" in self._grid:
                #     if "et" in self._grid["_ppc"]:
                #         self.comp_time += self._grid["_ppc"]["et"]
                # if self._grid.res_gen.isnull().values.any():
                #     # TODO see if there is a better way here -> do not handle this here, but rather in Backend._next_grid_state
                #     # sometimes pandapower does not detect divergence and put Nan.
                #     raise pdp.powerflow.LoadflowNotConverged("Divergence due to Nan values in res_gen table.")

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

                # TODO check how to handle

                # if not is_dc:
                #     if not np.all(np.isfinite(self.load_v)):
                #         # TODO see if there is a better way here
                #         # some loads are disconnected: it's a game over case!
                #         raise pp.powerflow.LoadflowNotConverged("Isolated load")
                # else:
                #     # fix voltages magnitude that are always "nan" for dc case
                #     # self._grid.res_bus["vm_pu"] is always nan when computed in DC
                #     self.load_v[:] = self.load_pu_to_kv  # TODO
                #     # need to assign the correct value when a generator is present at the same bus
                #     # TODO optimize this ugly loop
                #     for l_id in range(self.n_load):
                #         if self.load_to_subid[l_id] in self.gen_to_subid:
                #             ind_gens = np.where(
                #                 self.gen_to_subid == self.load_to_subid[l_id]
                #             )[0]
                #             for g_id in ind_gens:
                #                 if (
                #                         self._topo_vect[self.load_pos_topo_vect[l_id]]
                #                         == self._topo_vect[self.gen_pos_topo_vect[g_id]]
                #                 ):
                #                     self.load_v[l_id] = self.prod_v[g_id]
                #                     break


                self.p_or[:] = self._aux_get_line_info("p1", "p1", "p1")
                self.q_or[:] = self._aux_get_line_info("q1", "q1", "q1")
                self.v_or[:] = self._grid.get_buses()['v_mag'][self._grid.get_branches()['bus1_id'].values].values
                self.a_or[:] = self._aux_get_line_info("i1", "i1", "i1")
                self.theta_or[:] = self._grid.get_buses()['v_angle'][self._grid.get_branches()['bus1_id'].values].values
                self.a_or[~np.isfinite(self.a_or)] = 0.0
                self.v_or[~np.isfinite(self.v_or)] = 0.0

                self.p_ex[:] = self._aux_get_line_info("p2", "p2", "p2")
                self.q_ex[:] = self._aux_get_line_info("q2", "q2", "q2")
                self.v_ex[:] = self._grid.get_buses()['v_mag'][self._grid.get_branches()['bus2_id'].values].values
                self.a_ex[:] = self._aux_get_line_info("i2", "i2", "i2")
                self.theta_ex[:] = self._grid.get_buses()['v_angle'][self._grid.get_branches()['bus2_id'].values].values
                self.a_ex[~np.isfinite(self.a_ex)] = 0.0
                self.v_ex[~np.isfinite(self.v_ex)] = 0.0

                #TODO check the lines below to integrate them properly

                # # it seems that pandapower does not take into account disconencted powerline for their voltage
                # self.v_or[~self.line_status] = 0.0
                # self.v_ex[~self.line_status] = 0.0
                # self.v_or[:] *= self.lines_or_pu_to_kv
                # self.v_ex[:] *= self.lines_ex_pu_to_kv

                # # see issue https://github.com/rte-france/Grid2Op/issues/389
                # self.theta_or[~np.isfinite(self.theta_or)] = 0.0
                # self.theta_ex[~np.isfinite(self.theta_ex)] = 0.0
                #
                # self._nb_bus_before = None
                # self._grid._ppc["gen"][self._iref_slack, 1] = 0.0
                #
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

                if res[0].status == ppow._pypowsybl.LoadFlowComponentStatus.FAILED:
                    return False
                else:
                    return True

        except BackendError as exc_:
            # of the powerflow has not converged, results are Nan
            self._reset_all_nan()
            msg = exc_.__str__()
            return False, DivergingPowerFlow(f'powerflow diverged with error :"{msg}"')

    def get_line_status(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        As all the functions related to powerline, pandapower split them into multiple dataframe (some for transformers,
        some for 3 winding transformers etc.). We make sure to get them all here.
        """
        return self.line_status

    def _get_line_status(self):
        connected_1_lines = self._grid.get_lines()['connected1']
        connected_2_lines = self._grid.get_lines()['connected2']
        line_connected = connected_1_lines.values & connected_2_lines.values

        connected_1_2_transfo = self._grid.get_2_windings_transformers()['connected1']
        connected_2_2_transfo = self._grid.get_2_windings_transformers()['connected2']
        transfo_2_connected = connected_1_2_transfo.values & connected_2_2_transfo.values

        connected_1_3_transfo = self._grid.get_3_windings_transformers()['connected1']
        connected_2_3_transfo = self._grid.get_3_windings_transformers()['connected2']
        transfo_3_connected = connected_1_3_transfo.values & connected_2_3_transfo.values

        return np.concatenate(
            (
                line_connected,
                transfo_2_connected,
                transfo_3_connected
        )
        ).astype(dt_bool)

    def get_topo_vect(self):
        return self._topo_vect

    def _get_topo_vect(self):
        res = np.full(self.dim_topo, fill_value=np.NaN, dtype=dt_int)

        line_status = self.get_line_status()

        i = 0
        for row in self._grid.get_lines()[["bus1_id", "bus2_id"]].values:
            bus_or_id = self.map_sub[row[0]]
            bus_ex_id = self.map_sub[row[1]]

            if line_status[i]:
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
        for row in self._grid.get_2_windings_transformers()[["bus1_id", "bus2_id"]].values:
            bus_or_id = self.map_sub[row[0]]
            bus_ex_id = self.map_sub[row[1]]

            j = i + nb

            if line_status[j]:
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


        # For 3 windings transfo
        i = 0
        for row in self._grid.get_3_windings_transformers()[["bus1_id", "bus2_id"]].values:
            bus_or_id = self.map_sub[row[0]]
            bus_ex_id = self.map_sub[row[1]]

            if j is not None:
                k = j + i
            else:
                k = nb + i
            if line_status[k]:
                res[self.line_or_pos_topo_vect[k]] = (
                    1 if bus_or_id == self.line_or_to_subid[j] else 2
                )
                res[self.line_ex_pos_topo_vect[k]] = (
                    1 if bus_ex_id == self.line_ex_to_subid[j] else 2
                )
            else:
                res[self.line_or_pos_topo_vect[k]] = -1
                res[self.line_ex_pos_topo_vect[k]] = -1
            i += 1

        i = 0
        for bus_id in self._grid.get_generators()["bus_id"].values:
            res[self.gen_pos_topo_vect[i]] = 1 if self.map_sub[bus_id] == self.gen_to_subid[i] else 2
            i += 1

        i = 0
        for bus_id in self._grid.get_loads()["bus_id"].values:
            res[self.load_pos_topo_vect[i]] = (
                1 if self.map_sub[bus_id] == self.load_to_subid[i] else 2
            )
            i += 1


        if self.n_storage:
            # storage can be deactivated by the environment for backward compatibility
            i = 0
            for bus_id in self._grid.get_batteries()["bus_id"].values:
                status = self._grid.get_batteries()["connected"].values[i]
                if status:
                    res[self.storage_pos_topo_vect[i]] = (
                        1 if self.map_sub[bus_id] == self.storage_to_subid[i] else 2
                    )
                else:
                    res[self.storage_pos_topo_vect[i]] = -1
                i += 1

        return res

    def storages_info(self):
        return (
                self.storage_p,
                self.storage_q,
                self.storage_v,
        )

    def _storages_info(self):
        if self.n_storage:
            # this is because we support "backward comaptibility" feature. So the storage can be
            # deactivated from the Environment...
            p_storage = self._grid.get_batteries()["p"].values.astype(dt_float)
            q_storage = self._grid.get_batteries()["q"].values.astype(dt_float)
            v_storage = self._grid.get_buses()['v_mag'][self._grid.get_batteries()['bus_id']].values.astype(dt_float)
            theta_storage = self._grid.get_buses()['v_angle'][self._grid.get_batteries()['bus_id']].values.astype(dt_float)

        else:
            p_storage = np.zeros(shape=0, dtype=dt_float)
            q_storage = np.zeros(shape=0, dtype=dt_float)
            v_storage = np.zeros(shape=0, dtype=dt_float)
            theta_storage = np.zeros(shape=0, dtype=dt_float)
        return p_storage, q_storage, v_storage, theta_storage

    def generators_info(self):
        return (
            self.prod_p,
            self.prod_q,
            self.prod_v,
        )

    def _gens_info(self):
        prod_p = self._grid.get_generators()["p"].values.astype(dt_float)
        prod_q = self._grid.get_generators()["q"].values.astype(dt_float)
        prod_v = self._grid.get_buses()['v_mag'][self._grid.get_generators()['bus_id']].values.astype(dt_float)
        prod_theta = self._grid.get_buses()['v_angle'][self._grid.get_generators()['bus_id']].values.astype(dt_float)


        #TODO understand if the same problem occurs in powsybl

        # if self._iref_slack is not None:
        #     # slack bus and added generator are on same bus. I need to add power of slack bus to this one.
        #
        #     # if self._grid.gen["bus"].iloc[self._id_bus_added] == self.gen_to_subid[self._id_bus_added]:
        #     if "gen" in self._grid._ppc["internal"]:
        #         prod_p[self._id_bus_added] += self._grid._ppc["internal"]["gen"][
        #             self._iref_slack, 1
        #         ]
        #         prod_q[self._id_bus_added] += self._grid._ppc["internal"]["gen"][
        #             self._iref_slack, 2
        #         ]

        return prod_p, prod_q, prod_v, prod_theta
    def loads_info(self):
        return (
            self.load_p,
            self.load_q,
            self.load_v
        )

    def _loads_info(self):
        load_p = self._grid.get_loads()["p"].values.astype(dt_float)
        load_q = self._grid.get_loads()["q"].values.astype(dt_float)
        load_v = self._grid.get_buses()['v_mag'][self._grid.get_loads()['bus_id'].values].values.astype(dt_float)
        load_theta = self._grid.get_buses()['v_angle'][self._grid.get_loads()['bus_id'].values].values.astype(dt_float)
        return load_p, load_q, load_v, load_theta

    def lines_or_info(self):
        return(
            self.p_or,
            self.q_or,
            self.v_or,
            self.a_or
        )

    def lines_ex_info(self):
        return (
            self.p_ex,
            self.q_ex,
            self.v_ex,
            self.a_ex
        )

    def get_theta(self):
        return (
            self.theta_or,
            self.theta_ex,
            self.load_theta,
            self.gen_theta,
            self.storage_theta,
        )

    def assert_grid_correct(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is done as it should be by the Environment
        """
        super().assert_grid_correct()

    def _aux_get_line_info(self, colname1, colname2, colname3):
        res = np.concatenate(
            (
                self._grid.get_lines()[colname1].values,
                self._grid.get_2_windings_transformers()[colname2].values,
                self._grid.get_3_windings_transformers()[colname3].values,
            )
        )
        return res

    def sub_from_bus_id(self, bus_id):
        #TODO check that the function is doing what we want
        if bus_id >= self._number_true_line:
            print(bus_id)
            print(bus_id - self._number_true_line)
            return bus_id - self._number_true_line
        return bus_id

    def get_nb_active_bus(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

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
        L = []
        for elem in grid.get_voltage_levels().index:
            for bus_id in grid.get_bus_breaker_topology(voltage_level_id=elem).buses.index:
                L.append(bus_id)
        return L


    def _double_buses(self):
        df = self._grid.get_buses()
        L = []
        for elem in self._grid.get_voltage_levels().index:
            for bus_id in self._grid.get_bus_breaker_topology(voltage_level_id=elem).buses.index:
                L.append(bus_id)
        L_voltage_id = df['voltage_level_id'].to_list()
        for i in range(len(L)):
            self._grid.create_buses(id=L[i] + '_bis', voltage_level_id=L_voltage_id[i], name=df['name'][i])
