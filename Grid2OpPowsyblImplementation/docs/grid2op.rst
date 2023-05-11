.. module:: grid2op
.. _grid2op-module:

Grid2Op module
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

The grid2op module allows to model sequential decision making on a powergrid.

It is modular in the sense that it allows to use different powerflow solver (denoted as "Backend").
It proposes an internal representation
of the data that can be feed to powergrids and multiple classes to specify how it's done.

For example, it is possible to use an "action" to set the production value of some powerplant. But we
also know that it's not possible to do this for every powerplant (for example, asking a windfarm to produce more
energy is not possible: the only way would be to increase the speed of the wind). It is possible to implement
these kind of restrictions in this "game like" environment.

Today, the main usage of this platform is to serve as a computation engine for the `L2RPN <www.l2rpn.chalearn.com>`_
competitions.

This platform is still under development. If you notice a bug, let us know with a github issue at
`Grid2Op <https://github.com/rte-france/Grid2Op>`_

.. note:: Grid2op do not model any object on the powergrid. It has no internal modeling of the equations of the
    grids, or what kind of solver you need to adopt.

    On the other hand, grid2op aims at representing the grid in a relatively "high level" point of view: it knows
    which "elements" are connected to which, which is the production of this or that generators or how much power
    is consumed by this load.

    But under no circumstances, grid2op will expose some specific modeling of a powergrid. Such modeling are
    only made in the Backend.

    A somewhat relatively "accurate" view of what grid2op is to answer questions such as "if I had put a sensor at this
    location - *eg* right next to a powerplant - what would this sensor would have given ? It then takes care
    of exporting these data to a "format" to the entities acting on the grid.


Objectives
-----------
The primary goal of grid2op is to model decision making process in power systems. Indeed, we believe that developing
new flexibilities on the grid would make the
"energy transition" an easier, less costly process.

It allows fast development of new "methods" that are able to "take decisions" on a powergrid and assess how
well these controllers perform (see section `Controlling the grid`_ for more information about the "controls") .

Thanks to a careful separation between:

- the data used to represent how the powergrid is evolving (represented by the `Chronics`)
- the solver that is able to compute the state of the grid (represented by the `Backend`)
- the controller / agent that takes action on the powergrid (represented by the `Agent`)

All bound together thanks to the :class:`grid2op.Environment` module.

Grid2op attempts also to make the development of new control methods as easy as possible: it is relatively simple
to generate data and train agent on them and to use a fast (but less precise powerflow) while trying
to develop new state of the art methods. While still being usable in a "closer to reality" setting where data
can come from real grid state that happened in the past and the solver is as accurate as possible. You can switch
from one to the other almost effortlessly.

For a more detailed description, one can consult the
`Reinforcement Learning for Electricity Network Operation <https://arxiv.org/abs/2003.07339>`_
paper.

Though grid2op has been primarily developed for the L2RPN competitions series, it is more general. Its modularity
can also help developing and benchmarking new powerflow solvers for example.

Controlling the grid
--------------------
Modeling all what happens in the powergrid would be an extremely difficult task. Grid2op focusing on controls
that could be done today by a human (happening with **a frequency of approximately the minute**). It does not
aim at simulation really high frequency control that are often automatic today. That being said, such controls
can be taken into account by grid2op if the :class:`grid2op.Backend` allows it.

The main focus of grid2op is to make easy to use of **the topology** to control the flows of the grid.
In real time, it is possible to reconfigure the "topology" of the grid (you can think about it
by the action on changing the graph of the power network). Such modifications are highly non linear
and can have a really counter intuitive impact and we believe they are under used by industry and are under studied
by academics at the moment
(feel free to visit the notebooks `0_Introduction.ipynb`,
`0_SmallExample.ipynb` or the `IEEE BDA Tutorial Series.ipynb` of the official
`grid2op github repository <https://github.com/BDonnot/Grid2Op/tree/master/getting_started>`_ for more information)

Along with the topology, grid2op allows easily to manipulate (and thus control):

- the voltages: by manipulating shunts, or by changing the setpoint value of the generators
- the active generation: by the use of the "redispatching" action.
- the storage units (batteries or pumped storage) that allows to produce some energy / absorb some energy from the
  powergrid when needed.

Other "flexibilities" (ways to act on the grid) are coming soon (-:

.. note:: We wanted to emphasize the particularity of the problem proposed in grid2Op.
    Indeed, the objective is to act on a graph (observation space = a graph, action space = modification of this graph).

    As opposed to many graph related problems addressed in the literature, we do not try to find some properties of a
    dataset represented as one (or many) graph(s).

    Controlling a powergrid rather means to find a graph that meets some properties (**eg** all weights on all
    edges **aka** the flows on the powerlines, must be bellow some threshold **aka** the thermal limits - **NB** a
    solver uses some physical laws to compute these "weights" from the amount of power produced / absorbed in
    different part of the grid where generators and loads are connected).

What is modeled in an grid2op environment
-----------------------------------------
The simulator is able to emulate a power grid (of any size or characteristics) subject to a set of
temporal injections (productions and consumptions) or maintenance / hazards for discretized
time-steps (usually there is the equivalent of *5* minutes between two consective steps).

Say a powergrid is represented as a graph with:

- the edges being the powerlines (and transformers)
- the nodes being the "bus" (a bus is the power system terminology to denotes the "things" (aka nodes) that are
  connected by the edges)

.. note:: Grid2op does not explicitly model the "graph" of the grid as a "graph" structure. For performances, it is
    represented as a vector, as explained in paragraph ":ref:`topology-pb-explained`". To be exhaustive, the way to
    map this graph to this vector is explained in the page ":ref:`create-backend-module`" (though this page is
    really detailed and has too much information for most grid2op usage).

    Some functions have been coded to retrieve the state, as a "graph" (more precisely a square matrix). These methods
    are described in the section ":ref:`observation_module_graph`" of the Observation module.

This graph has some properties:

- some buses are labeled "generators" that produces a certain amount of power
- some buses are labeled "loads" that consumes a certain amount of power  (**NB** a bus can be both a generator
  and a load, in this case both the production and the demand should be met at his node)
- all edges have some  "weights": some physical laws (*eg* conservation of energy or more specifically
  `Kirchoff Circuits Laws`), that cannot be altered (and are computed by the `Backend`), induced some flows on
  the powerline that can be represented as "weights" on this graph
- it is dynamic: at different steps, the graph can be different, for example, it is possible to have a "node" with
  load 1, load 2, line 1 and line 2 and a given step, and to "split" this node in two to have, at another step
  load 1 and line 2 on a "node" and "load 2" and "line 1" on a different node (and the other way around).

This graph has some constraints:

- the total generation (sum of production of all generator) should be exactly equal to the
  total demand (sum of consumption of all loads) and the power losses (due to the heating of the powerlines for
  example)
- the generators should always be connected to the grid, otherwise this is a blackout
- the loads should always be connected to the grid, otherwise this is a blackout
- the graph of the grid should be `connected` (made of one unique connex component): otherwise the condition number
  1 above (sum production = sum load + sum losses) will not be met in each of the independant subgraph, most likely.
- there exist a solution to the `Kirchoff Circuits Laws`

For more information on this "graph" and the way to retrieve it
in different format, you can consult the page :ref:`gridgraph-module` of the documentation.

The whole grid2op ecosystem aims at modeling the evolution of a "controller" that is able to make sure the
"graph of grid", at all time meets all the constraints.

More concretely a grid2op environment models "out of the box":

- the mechanism to "implement" a control on the grid (with a dedicated `action` module) that can be used by any
  `Agent`, which takes some decisions to maintain the grid in security
- time series of loads and productions: which represents the evolution of the power injected / withdrawn
  at each bus of the grid, at any time (**NB** the `Agent` do not see the future, it means that it cannot have an
  exact value for each of the loads in the future, but can only observe the current sate)
- a mechanism (that can be implemented using different solver) to compute the flows based on the injections (which
  among of power is produced at each nodes) and the topology (graph of the grid)
- the automatic disconnection of powerlines if there are on overflow for too long (known as "time overcurrent (TOC)" see
  this article for more information
  `overcurrent <https://en.wikipedia.org/wiki/Power_system_protection#Overload_and_back-up_for_distance_(overcurrent)>`_ )
  Conceptually this means the environment remember for how long a powergrid is in "overflow" and disconnects it
  if needed. **NB** This is an **emulation** of what happen on the grid, in case you use a Backend that do not have
  this feature (for example if you use static / steady state powerflow). This emulation might not be necessary (and
  less "realistic" if you use a time domain simulator)
- the disconnection of powerlines if the overflow is too high (known as "instantaneous overcurrent" see the same
  wikipedia article). This means from one step to another, a given powerline can be disconnected if too much
  flow goes through it. **NB** This is an **emulation** of what happen on the grid, in case you use a Backend that do not have
  this feature (for example if you use static / steady state powerflow). This emulation might not be necessary (and
  less "realistic" if you use a time domain simulator)
- the maintenance operations: if there is a planned maintenance, the environment is able to disconnect a powerline
  for a given amount of steps and preventing its reconnection. There are information about such planned event
  that are given to the controller.
- hazards / unplanned outages / attacks: another issue on power system is the fact that sometimes, some powerline
  get disconnected in a non planned manner. For example, a tree can fall on a powerline, the grid might suffer
  a cyber attack etc. This can also be modeled by grid2op.
- prevent the action on some powerlines: whether it is to model the fact in reality it is not possible to always
  act on the same equipment or because some powerline are out of service (because of an attack, a maintenance
  or because it needs to be repaired), grid2op can model the impossibility
  of acting on a given powerline
- prevent the action on some substations: for the same reasons, sometimes you cannot act on given part of
  the network, preventing you to do some topological actions.
- voltage control: though it is not the main focus of the current platform, grid2op can model automatons that
  can take voltage corrective measures (in the near future we think of adding some protection monitoring
  voltage violation too).
- non violation of generator physical constraints: in real life, generator cannot produce too little nor too much
  (we speak about `gen_pmin` and `gen_pmax`) nor their production can vary too much between consecutive
  steps (this is called `gen_max_ramp_down` and `gen_max_ramp_up`)
- stops the game if the grid is in a too bad shape. This can happen if a load or a generator has been disconnected,
  or if some part of the grid is "islanded" (the graph representing the power network is not connex) or if there is
  no feasible solution to the power system equations

Here are a summary of the main modules:

=============================  =========================================================================================
Module Name                    Main usage
=============================  =========================================================================================
:class:`grid2op.Environment`   Implements all the mechanisms described above
:class:`grid2op.Chronics`      In charge of feeding the data (loads, generations, planned maintenance, etc.) to the Environment
:class:`grid2op.Backend`       Carries out the computation of the powergrid state
:class:`grid2op.Agent`         The controller, in charge of managing the safety of the grid
:class:`grid2op.Action`        The control send by the Agent to the Environment
:class:`grid2op.Observartion`  The information sent by the Environment to the Agent, represents the powergrid state as seen by the Agent
:class:`grid2op.Opponent`      Is present to model the unplanned disconnections of powerline
:class:`grid2op.Rules`         Computes whether or not an action is "legal" at a given time step
:class:`grid2op.Parameters`    Store the parameters that defines for example, on which case an action is legal, or how long a powerline can stay on overflow etc.
=============================  =========================================================================================

Properties of this environments
-------------------------------
The grid2op environments have multiple shared properties:

- highly constrained environments: these environments obey physical laws. You cannot directly choose how much
  power flow on a given powerline, what you can do it choosing the "graph" of the power network and (under some
  constraints) the production of each generators. Knowing these information at any time steps, the powergrid state
  must satisfy the `Kirchhoff's circuit laws <https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws>`_ .
- stochastic environments: in all environment, you don't know the future, which makes it a "Partially
  Observable" environments (if you were in a maze, you would **not** see "from above" but rather see "at the first
  person": only seeing in front of you).
  Environments can be "even more stochastic" if there are hazards or even adversarial: a malicious agent can take
  attacks targeted to endanger your policy.
- with both **continuous and discrete observation space**: some part of the observation are continuous (for example
  the amount of flow on a given powerline, or the production of this generator) and some are discrete (
  for example the status - connected / disconnected - of a powerline, or how long this powerline
  has been in overflow etc.)
- with **both continuous and discrete action space**: the preferred type of action is the topology, which is
  represented as a discrete type of action (*eg* you can either connect / disconnect a powerline) but there exist
  also some continuous actions (for example you can adjust in real time the production of a set of generators)
- dynamic graph manipulation: power network can be modeled as graphs. In these environments both the observation
  **and the action** are focused on graph. The observation contains the complete state of the grid, including
  the "topology" (you can think of it a its graph) and actions are focused on adapting this graph to make
  the grid as robust and secure as possible. **NB** As opposed to most problem in the literature, where
  you need to find some properties (label of of the edges or the nodes, etc.) in grid2op you need
  to find a graph that meets some properties: find a graph that meets constraints on its edges and its nodes.
- strong emphasis on **safety** and **security**: power system are highly critical system (who would want to
  short circuit a powerplant? Or causing a blackout preventing an hospital to cure the patients?) and as such it is
  critical that the controls keep the powergrid safe in all circumstances.

Disclaimer
-----------
Grid2op is a research testbed platform, it shall not be use in "production" for any kind of application.


Going further
--------------
To get started into the grid2op ecosystem, we made a set of notebooks
that are available, without any installation thanks to
`Binder <https://mybinder.org/v2/gh/rte-france/Grid2Op/master>`_ . Feel free to visit the "getting_started" page for
more information and a detailed tour about the issue that grid2op tries to address.

.. note:: As of writing (december 2020) most of these notebooks focus on the "agent" part of grid2op. We would welcome
    any contribution to better explain the other aspect of this platform.
