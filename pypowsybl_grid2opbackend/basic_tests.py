# Copyright (c) 2023, Artelys (https://www.artelys.com/)
# @author Rémi Tschupp <remi.tschupp@artelys.com>
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of pypowsybl-grid2opbackend. It is mostly inspired by the development of the several backends from
# Grid2op framework. Some part of codes have been paste/copy.

import unittest
from pypowsybl_grid2opbackend.Backend.powsybl_backend import PowsyblBackend


from grid2op._create_test_suite import create_test_suite

def this_make_backend(self, detailed_infos_for_cascading_failures=False):
    return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )
add_name_cls = "test_pypowsyblBackend"

res = create_test_suite(make_backend_fun=this_make_backend,
                        add_name_cls=add_name_cls,
                        add_to_module=__name__,
                        extended_test=False,  # for now keep `extended_test=False` until all problems are solved
                        )

# and run it with `python -m unittest basic_test.py`
if __name__ == "__main__":
    unittest.main()