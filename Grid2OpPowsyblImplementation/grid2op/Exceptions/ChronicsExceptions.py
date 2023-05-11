# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions.Grid2OpException import Grid2OpException


# Chronics
class ChronicsError(Grid2OpException):
    """
    Base class of all error regarding the chronics and the gridValue (see :class:`grid2op.ChronicsHandler.GridValue` for
    more information)
    """

    pass


class ChronicsNotFoundError(ChronicsError):
    """
    This exception is raised where there are no chronics folder found at the indicated location.
    """

    pass


class InsufficientData(ChronicsError):
    """
    This exception is raised where there are not enough data compare to the size of the episode asked.
    """

    pass
