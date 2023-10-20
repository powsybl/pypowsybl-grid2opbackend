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
from src.Backend.powsybl_backend import PowsyblBackend
from src.tests.helper_path_test import HelperTests
from src.tests.base_backend_test_powsybl import BaseTestLoadingCase
from src.tests.base_backend_test_powsybl import BaseTestLoadingBackendFunc
from src.tests.base_backend_test_powsybl import BaseTestTopoAction
import warnings
warnings.simplefilter("error")

test_dir = Path(__file__).parent.absolute()
implementation_dir = os.fspath(test_dir.parent.absolute())
data_dir = os.path.abspath(os.path.join(implementation_dir, "data_test"))
PATH_DATA_TEST_PYPOW = data_dir


class TestLoadingCase(HelperTests, BaseTestLoadingCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PowsyblBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

    def get_path(self):
        return PATH_DATA_TEST_PYPOW

    def get_casefile(self):
        return "grid.json"


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
        return PATH_DATA_TEST_PYPOW

    def get_casefile(self):
        return "grid.json"


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
        return PATH_DATA_TEST_PYPOW

    def get_casefile(self):
        return "grid.json"

if __name__ == "__main__":
    unittest.main()

