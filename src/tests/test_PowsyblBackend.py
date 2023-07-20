# Copyright (c) 2023, Artelys (https://www.artelys.com/)
# @author Rémi Tschupp <remi.tschupp@artelys.com>
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import numpy as np
import os
from grid2op import make
from pathlib import Path
from Backend.PowsyblBackend import PowsyblBackend

from grid2op.tests.helper_path_test import PATH_DATA_TEST_PP, PATH_DATA_TEST
from grid2op.tests.helper_path_test import HelperTests
from BaseBackendTestPyPowsybl import BaseTestNames
from BaseBackendTestPyPowsybl import BaseTestLoadingCase
from BaseBackendTestPyPowsybl import BaseTestLoadingBackendFunc
from BaseBackendTestPyPowsybl import BaseTestTopoAction

# from grid2op.tests.BaseBackendTest import BaseTestNames
# from grid2op.tests.BaseBackendTest import BaseTestLoadingCase
# from grid2op.tests.BaseBackendTest import BaseTestLoadingBackendFunc
# from grid2op.tests.BaseBackendTest import BaseTestTopoAction
from grid2op.tests.BaseBackendTest import BaseTestEnvPerformsCorrectCascadingFailures
from grid2op.tests.BaseBackendTest import BaseTestChangeBusAffectRightBus
from grid2op.tests.BaseBackendTest import BaseTestShuntAction
from grid2op.tests.BaseBackendTest import BaseTestResetEqualsLoadGrid
from grid2op.tests.BaseBackendTest import BaseTestVoltageOWhenDisco
from grid2op.tests.BaseBackendTest import BaseTestChangeBusSlack
from grid2op.tests.BaseBackendTest import BaseIssuesTest
from grid2op.tests.BaseBackendTest import BaseStatusActions
from grid2op.tests.BaseBackendTest import BaseTestStorageAction

test_dir = Path(__file__).parent.absolute()
implementation_dir = os.fspath(test_dir.parent.absolute())
data_dir = os.path.abspath(os.path.join(implementation_dir, "data_test"))
PATH_DATA_TEST_PYPOW = data_dir

PATH_DATA_TEST_INIT = PATH_DATA_TEST
PATH_DATA_TEST = PATH_DATA_TEST_PP

import warnings

warnings.simplefilter("error")


class TestLoadingCase(HelperTests, BaseTestLoadingCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_path(self):
        return PATH_DATA_TEST #PATH_DATA_TEST_PYPOW

    def get_casefile(self):
        return "test_case14.json" #"test_case14.xiidm" #"case14_realistic_test.mat"


# class TestNames(HelperTests, BaseTestNames):
#     def make_backend(self, detailed_infos_for_cascading_failures=False):
#         return PowsyblBackend(
#             detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
#         )
#
#     def get_path(self):
#         return PATH_DATA_TEST



class TestLoadingBackendFunc(HelperTests, BaseTestLoadingBackendFunc):
    def setUp(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_path(self):
        return PATH_DATA_TEST #PATH_DATA_TEST_PYPOW

    def get_casefile(self):
        return "test_case14.json"  #"test_case14.xiidm"


class TestTopoAction(HelperTests, BaseTestTopoAction):
    def setUp(self):
        BaseTestTopoAction.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestTopoAction.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_path(self):
        return PATH_DATA_TEST #PATH_DATA_TEST_PYPOW

    def get_casefile(self):
        return "test_case14.json"  #"test_case14.xiidm"

if __name__ == "__main__":
    unittest.main()

