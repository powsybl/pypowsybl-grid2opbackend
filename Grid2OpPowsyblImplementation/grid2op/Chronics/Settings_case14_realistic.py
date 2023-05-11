# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This file contains the settings (path to the case file, chronics converter etc.) that allows to make a simple
environment with a powergrid of only 5 buses, 3 laods, 2 generators and 8 powerlines.
"""
import os
import numpy as np
from pathlib import Path

file_dir = Path(__file__).parent.absolute()
grid2op_root = file_dir.parent.absolute()
grid2op_root = str(grid2op_root)
dat_dir = os.path.abspath(os.path.join(grid2op_root, "data"))
case_dir = "rte_case14_realistic"
grid_file = "grid.json"

case14_real_CASEFILE = os.path.join(dat_dir, case_dir, grid_file)
case14_real_CHRONICSPATH = os.path.join(dat_dir, case_dir, "chronics")

case14_real_TH_LIM = np.array(
    [
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
    ]
)
