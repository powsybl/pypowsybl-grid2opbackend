# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions.Grid2OpException import Grid2OpException


# Exception Runner is used twice, not possible on windows / macos due to the way multiprocessing works
class UsedRunnerError(Grid2OpException):
    """
    This exception indicate that runner (object of :class:`grid2op.Runner.Runner`) has already been used.

    This behaviour is not supported on windows / macos given the way the Multiprocessing package works (spawning
    a process where grid2op objects are made is not completly supported at the moment).

    The best solution is to recreate a runner, and then use this new one.
    """

    pass
