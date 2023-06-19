from __future__ import annotations  # Necessary for type alias like _DataFrame to work with sphinx
import copy

from os import PathLike as _PathLike
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Union as _Union

import pypowsybl._pypowsybl as _pp
from numpy.typing import ArrayLike as _ArrayLike
from pandas import DataFrame as _DataFrame
from pypowsybl.network import Network, _path_to_str
from pypowsybl.report import Reporter as _Reporter

# Type definitions
if _TYPE_CHECKING:
    ParamsDict = _Optional[_Dict[str, str]]
    PathOrStr = _Union[str, _PathLike]


class SortedNetwork(Network):
    def __init__(self, *args, **kwargs):
        super(SortedNetwork, self).__init__(*args, **kwargs)
        self._loads_index = super(SortedNetwork, self).get_loads().index

    def deepcopy(self):
        network_copy = copy.deepcopy(self)
        network_copy._loads_index = self._loads_index
        return network_copy

    def get_loads(self, *args, **kwargs):
        if self._loads_index is None:
            return super(SortedNetwork, self).get_loads(*args, **kwargs)
        else:
            return super(SortedNetwork, self).get_loads(*args, **kwargs).loc[self._loads_index, :]


def load(file: _Union[str, _PathLike], parameters: _Dict[str, str] = None, reporter: _Reporter = None) -> Network:
    """
    Load a network from a file. File should be in a supported format.

    Basic compression formats are also supported (gzip, bzip2).

    Args:
       file:       path to the network file
       parameters: a dictionary of import parameters
       reporter:   the reporter to be used to create an execution report, default is None (no report)

    Returns:
        The loaded network

    Examples:

        Some examples of file loading, including relative or absolute paths, and compressed files:

        .. code-block:: python

            network = pp.network.load('network.xiidm')
            network = pp.network.load('/path/to/network.xiidm')
            network = pp.network.load('network.xiidm.gz')
            network = pp.network.load('network.uct')
            ...
    """
    file = _path_to_str(file)
    if parameters is None:
        parameters = {}
    return SortedNetwork(_pp.load_network(file, parameters,
                                    None if reporter is None else reporter._reporter_model))  # pylint: disable=protected-access


def _create_network(name: str, network_id: str = '') -> Network:
    return SortedNetwork(_pp.create_network(name, network_id))
