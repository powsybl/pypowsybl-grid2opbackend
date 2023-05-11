# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from gym import spaces
import warnings
import numpy as np

from grid2op.Environment import (
    Environment,
    MultiMixEnvironment,
    BaseMultiProcessEnvironment,
)
from grid2op.Action import BaseAction, ActionSpace
from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Converter.Converters import Converter
from grid2op.gym_compat.base_gym_attr_converter import BaseGymAttrConverter
from grid2op.gym_compat.gym_space_converter import _BaseGymSpaceConverter


class GymActionSpace(_BaseGymSpaceConverter):
    """
    This class enables the conversion of the action space into a gym "space".

    Resulting action space will be a :class:`gym.spaces.Dict`.

    **NB** it is NOT recommended to use the sample of the gym action space. Please use the sampling (
    if availabe) of the original action space instead [if not available this means there is no
    implemented way to generate reliable random action]

    **Note** that gym space converted with this class should be seeded independently. It is NOT seeded
    when calling :func:`grid2op.Environment.Environment.seed`.

    Examples
    --------
    Converting an action space is fairly straightforward, though the resulting gym action space
    will depend on the original encoding of the action space.

    .. code-block:: python

        import grid2op
        from grid2op.Converter import GymActionSpace
        env = grid2op.make()

        gym_action_space = GymActionSpace(env)
        # and now gym_action_space is a `gym.spaces.Dict` representing the action space.
        # you can convert action to / from this space to grid2op the following way

        grid2op_act = env.action_space(...)
        gym_act = gym_action_space.to_gym(grid2op_act)

        # and the opposite conversion is also possible:
        gym_act = ... # whatever you decide to do
        grid2op_act = gym_action_space.from_gym(gym_act)

    **NB** you can use this `GymActionSpace` to  represent action into the gym format even if these actions
    comes from another converter, such as :class`IdToAct` or `ToVect` in this case, to get back a grid2op
    action you NEED to convert back the action from this converter. Here is a complete example
    on this (more advanced) usecase:

    .. code-block:: python

        import grid2op
        from grid2op.Converter import GymActionSpace, IdToAct
        env = grid2op.make()

        converted_action_space = IdToAct(env)
        gym_action_space = GymActionSpace(env=env, converter=converted_action_space)

        # and now gym_action_space is a `gym.spaces.Dict` representing the action space.
        # you can convert action to / from this space to grid2op the following way

        converter_act = ... # whatever action you want
        gym_act = gym_action_space.to_gym(converter_act)

        # and the opposite conversion is also possible:
        gym_act = ... # whatever you decide to do
        converter_act = gym_action_space.from_gym(gym_act)

        # note that this converter act only makes sense for the converter. It cannot
        # be digest by grid2op directly. So you need to also convert it to grid2op
        grid2op_act = IdToAct.convert_act(converter_act)

    """

    # deals with the action space (it depends how it's encoded...)
    keys_grid2op_2_human = {
        "prod_p": "prod_p",
        "prod_v": "prod_v",
        "load_p": "load_p",
        "load_q": "load_q",
        "_redispatch": "redispatch",
        "_set_line_status": "set_line_status",
        "_switch_line_status": "change_line_status",
        "_set_topo_vect": "set_bus",
        "_change_bus_vect": "change_bus",
        "_hazards": "hazards",
        "_maintenance": "maintenance",
        "_storage_power": "storage_power",
        "_curtail": "curtail",
        "_raise_alarm": "raise_alarm",
        "shunt_p": "_shunt_p",
        "shunt_q": "_shunt_q",
        "shunt_bus": "_shunt_bus",
    }
    keys_human_2_grid2op = {v: k for k, v in keys_grid2op_2_human.items()}

    def __init__(self, env, converter=None, dict_variables=None):
        """
        note: for consistency with GymObservationSpace, "action_space" here can be an environment or
        an action space or a converter
        """
        if dict_variables is None:
            dict_variables = {}
        if isinstance(
            env, (Environment, MultiMixEnvironment, BaseMultiProcessEnvironment)
        ):
            # action_space is an environment
            self.initial_act_space = env.action_space
            self._init_env = env
        elif isinstance(env, ActionSpace) and converter is None:
            warnings.warn(
                "It is now deprecated to initialize an Converter with an "
                "action space. Please use an environment instead."
            )
            self.initial_act_space = env
            self._init_env = None
        else:
            raise RuntimeError(
                "GymActionSpace must be created with an Environment of an ActionSpace (or a Converter)"
            )
        dict_ = {}
        # TODO Make sure it works well !
        if converter is not None and isinstance(converter, Converter):
            # a converter allows to ... convert the data so they have specific gym space
            self.initial_act_space = converter
            dict_ = converter.get_gym_dict()
            self.__is_converter = True
        elif converter is not None:
            raise RuntimeError(
                'Impossible to initialize a gym action space with a converter of type "{}" '
                "A converter should inherit from grid2op.Converter".format(
                    type(converter)
                )
            )
        else:
            self._fill_dict_act_space(
                dict_, self.initial_act_space, dict_variables=dict_variables
            )
            dict_ = self._fix_dict_keys(dict_)
            self.__is_converter = False
        _BaseGymSpaceConverter.__init__(self, dict_, dict_variables)

    def reencode_space(self, key, fun):
        """
        This function is used  to reencode the action space. For example, it can be used to scale
        the observation into values close to 0., it can also be used to encode continuous variables into
        discrete variables or the other way around etc.

        Basically, it's a tool that lets you define your own observation space (there is the same for
        the action space)

        Parameters
        ----------
        key: ``str``
            Which part of the observation space you want to study

        fun: :class:`BaseGymAttrConverter`
            Put `None` to deactivate the feature (it will be hided from the observation space)
            It can also be a `BaseGymAttrConverter`. See the example for more information.

        Returns
        -------
        self:
            The current instance, to be able to chain these calls

        Notes
        ------
        It modifies the observation space. We highly recommend to set it up at the beginning of your script
        and not to modify it afterwards

        'fun' should be deep copiable (meaning that if `copy.deepcopy(fun)` is called, then it does not crash

        If an attribute has been ignored, for example by :func`GymEnv.keep_only_obs_attr`
        or and is now present here, it will be re added in the final observation
        """
        if self._init_env is None:
            raise RuntimeError(
                "Impossible to reencode a space that has been initialized with an "
                "action space as input. Please provide a valid"
            )
        if self.__is_converter:
            raise RuntimeError(
                "Impossible to reencode a space that is a converter space."
            )

        my_dict = self.get_dict_encoding()
        if fun is not None and not isinstance(fun, BaseGymAttrConverter):
            raise RuntimeError(
                "Impossible to initialize a converter with a function of type {}".format(
                    type(fun)
                )
            )
        if key in self.keys_human_2_grid2op:
            key2 = self.keys_human_2_grid2op[key]
        else:
            key2 = key

        if fun is not None and not fun.is_init_space():
            if key2 in my_dict:
                fun.initialize_space(my_dict[key2])
            elif key in self.spaces:
                fun.initialize_space(self.spaces[key])
            else:
                raise RuntimeError(f"Impossible to find key {key} in your action space")
        my_dict[key2] = fun
        res = GymActionSpace(env=self._init_env, dict_variables=my_dict)
        return res

    def _fill_dict_act_space(self, dict_, action_space, dict_variables):
        # TODO what about dict_variables !!!
        for attr_nm, sh, dt in zip(
            action_space.attr_list_vect, action_space.shape, action_space.dtype
        ):
            if sh == 0:
                # do not add "empty" (=0 dimension) arrays to gym otherwise it crashes
                continue
            my_type = None
            shape = (sh,)
            if attr_nm in dict_variables:
                # case where the user specified a dedicated encoding
                if dict_variables[attr_nm] is None:
                    # none is by default to disable this feature
                    continue
                my_type = dict_variables[attr_nm].my_space
            elif dt == dt_int:
                # discrete action space
                if attr_nm == "_set_line_status":
                    my_type = spaces.Box(low=-1, high=1, shape=shape, dtype=dt)
                elif attr_nm == "_set_topo_vect":
                    my_type = spaces.Box(low=-1, high=2, shape=shape, dtype=dt)
            elif dt == dt_bool:
                # boolean observation space
                my_type = self._boolean_type(sh)
                # case for all "change" action and maintenance / hazards
            else:
                # continuous observation space
                low = float("-inf")
                high = float("inf")
                shape = (sh,)
                SpaceType = spaces.Box

                if attr_nm == "prod_p":
                    low = action_space.gen_pmin
                    high = action_space.gen_pmax
                    shape = None
                elif attr_nm == "prod_v":
                    # voltages can't be negative
                    low = 0.0
                elif attr_nm == "_redispatch":
                    # redispatch
                    low = -1.0 * action_space.gen_max_ramp_down
                    high = 1.0 * action_space.gen_max_ramp_up
                    low[~action_space.gen_redispatchable] = 0.0
                    high[~action_space.gen_redispatchable] = 0.0
                elif attr_nm == "_curtail":
                    # curtailment
                    low = np.zeros(action_space.n_gen, dtype=dt_float)
                    high = np.ones(action_space.n_gen, dtype=dt_float)
                    low[~action_space.gen_renewable] = 1.0
                    high[~action_space.gen_renewable] = 1.0
                elif attr_nm == "_storage_power":
                    # storage power
                    low = -1.0 * action_space.storage_max_p_prod
                    high = 1.0 * action_space.storage_max_p_absorb
                my_type = SpaceType(low=low, high=high, shape=shape, dtype=dt)

            if my_type is None:
                # if nothing has been found in the specific cases above
                my_type = self._generic_gym_space(dt, sh)

            dict_[attr_nm] = my_type

    def _fix_dict_keys(self, dict_: dict) -> dict:
        res = {}
        for k, v in dict_.items():
            res[self.keys_grid2op_2_human[k]] = v
        return res

    def from_gym(self, gymlike_action: spaces.dict.OrderedDict) -> object:
        """
        Transform a gym-like action (such as the output of "sample()") into a grid2op action

        Parameters
        ----------
        gymlike_action: :class:`gym.spaces.dict.OrderedDict`
            The action, represented as a gym action (ordered dict)

        Returns
        -------
        An action that can be understood by the given action_space (either a grid2Op action if the
        original action space was used, or a Converter)

        """
        if self.__is_converter:
            # case where the action space comes from a converter, in this case the converter takes the
            # delegation to convert the action to openai gym
            res = self.initial_act_space.convert_action_from_gym(gymlike_action)
        else:
            # case where the action space is a "simple" action space
            res = self.initial_act_space()
            for k, v in gymlike_action.items():
                internal_k = self.keys_human_2_grid2op[k]
                if internal_k in self._keys_encoding:
                    tmp = self._keys_encoding[internal_k].gym_to_g2op(v)
                else:
                    tmp = v
                res._assign_attr_from_name(internal_k, tmp)
        return res

    def to_gym(self, action: object) -> spaces.dict.OrderedDict:
        """
        Transform an action (non gym) into an action compatible with the gym Space.

        Parameters
        ----------
        action:
            The action (coming from grid2op or understandable by the converter)

        Returns
        -------
        gym_action:
            The same action converted as a OrderedDict (default used by gym in case of action space
            being Dict)

        """
        if self.__is_converter:
            gym_action = self.initial_act_space.convert_action_to_gym(action)
        else:
            # in that case action should be an instance of grid2op BaseAction
            assert isinstance(
                action, BaseAction
            ), "impossible to convert an action not coming from grid2op"
            # TODO this do not work in case of multiple converter,
            #  TODO this should somehow call tmp = self._keys_encoding[internal_k].g2op_to_gym(v)
            gym_action = self._base_to_gym(
                self.spaces.keys(),
                action,
                dtypes={k: self.spaces[k].dtype for k in self.spaces},
                converter=self.keys_human_2_grid2op,
            )
        return gym_action

    def close(self):
        if hasattr(self, "_init_env"):
            self._init_env = None  # this doesn't own the environment
