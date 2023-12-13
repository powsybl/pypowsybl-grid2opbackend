# Copyright (c) 2023, Artelys (https://www.artelys.com/)
# See Authors.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of pypowsybl-grid2opbackend. It is mostly inspired by the development of the several backends from
# Grid2op framework. Most parts of code have been paste/copy.

import unittest
import os
from pathlib import Path
from pypowsybl_grid2opbackend.Backend.powsybl_backend import PowsyblBackend
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestLoadingCase
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestLoadingBackendFunc
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestTopoAction
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestEnvPerformsCorrectCascadingFailures
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestChangeBusAffectRightBus
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestShuntAction
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestResetEqualsLoadGrid
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestVoltageOWhenDisco
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestChangeBusSlack
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseIssuesTest
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseStatusActions
from pypowsybl_grid2opbackend.tests.base_backend_test_powsybl import BaseTestStorageAction
import warnings
warnings.simplefilter("error")

test_dir = Path(__file__).parent.absolute()
implementation_dir = os.fspath(test_dir.parent.absolute())
data_dir = os.path.abspath(os.path.join(implementation_dir, os.path.join("data_test", "l2rpn_case14_sandbox_Pypowsybl")))
PATH_DATA_TEST_PYPOW = data_dir


class TestLoadingCase(BaseTestLoadingCase, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_path(self):
        return PATH_DATA_TEST_PYPOW

    def get_casefile(self):
        return "grid.json"


class TestLoadingBackendFunc(BaseTestLoadingBackendFunc, unittest.TestCase):
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
        return PATH_DATA_TEST_PYPOW

    def get_casefile(self):
        return "grid.json"


class TestTopoAction(BaseTestTopoAction, unittest.TestCase):
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
        return PATH_DATA_TEST_PYPOW

    def get_casefile(self):
        return "grid.json"


class TestEnvPerformsCorrectCascadingFailures(
    BaseTestEnvPerformsCorrectCascadingFailures, unittest.TestCase
):
    def setUp(self):
        BaseTestEnvPerformsCorrectCascadingFailures.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestEnvPerformsCorrectCascadingFailures.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_casefile(self):
        return "grid.json"

    def get_path(self):
        return PATH_DATA_TEST_PYPOW


class TestChangeBusAffectRightBus(BaseTestChangeBusAffectRightBus, unittest.TestCase):

    def setUp(self):
        BaseTestChangeBusAffectRightBus.setUp(self)
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_casefile(self):
        return "grid.json"

    def get_path(self):
        return PATH_DATA_TEST_PYPOW

# TODO work on shunt tests
# class TestShuntAction(BaseTestShuntAction, unittest.TestCase):
#
#     def setUp(self):
#         BaseTestShuntAction.setUp(self)
#
#     def make_backend(self, detailed_infos_for_cascading_failures=False):
#         return PowsyblBackend(
#             detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
#         )
#
#     def get_casefile(self):
#         return "grid.json"
#
#     def get_path(self):
#         return PATH_DATA_TEST_PYPOW

class TestResetEqualsLoadGrid(BaseTestResetEqualsLoadGrid, unittest.TestCase):
    def setUp(self):
        BaseTestResetEqualsLoadGrid.setUp(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_casefile(self):
        return "grid.json"

    def get_path(self):
        return PATH_DATA_TEST_PYPOW



class TestVoltageOWhenDisco(BaseTestVoltageOWhenDisco, unittest.TestCase):

    def setUp(self):
        BaseTestVoltageOWhenDisco.setUp(self)
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_casefile(self):
        return "grid.json"

    def get_path(self):
        return PATH_DATA_TEST_PYPOW


class TestChangeBusSlack(BaseTestChangeBusSlack, unittest.TestCase):
    def setUp(self):
        BaseTestChangeBusSlack.setUp(self)
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_casefile(self):
        return "grid.json"

    def get_path(self):
        return PATH_DATA_TEST_PYPOW


# class TestIssuesTest(BaseIssuesTest):
#     def make_backend(self, detailed_infos_for_cascading_failures=False):
#         return PowsyblBackend(
#             detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
#         )


class TestStatusAction(BaseStatusActions, unittest.TestCase):

    def setUp(self):
        BaseStatusActions.setUp(self)
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )
    def get_casefile(self):
        return "grid.json"

    def get_path(self):
        return PATH_DATA_TEST_PYPOW


if __name__ == "__main__":
    unittest.main()

