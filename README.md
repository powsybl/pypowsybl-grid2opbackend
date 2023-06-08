# pypowsybl-grid2opbackend integration (AIRGo project)


## Definition and comparison of objects between Pypowsybl and Grid2op

### Lines and transformers

#### Pypowsybl
In Pypowsybl there is a real distinction between lines, 2 winding transformers and 3 winding transformers.

#### Grid2op
In Grid2op all those objects are assimilated as lines. Nevertheless, Grid2op knows which of these lines are real powerlines 
and those that are not.


### Thermal limitation

#### Pypowsybl
In Pypowsybl thermal limitations are set for both buses of a line (two extremities).

#### Grid2op
In Grid2op thermal limitations are set for the entire line.

We decided to choose the smallest value of both extremities in pypowsybl and to give that information for the line thermal 
limitation in Grid2op. We also chose to take into account only permanent limit in current. By default if none information
is available in pypowsybl I set a huge limit which is equal to no limit.

### Substations

#### Pypowsybl
In Pypowsybl substations should contain at least one voltage level and busbar.

#### Grid2op
In Grid2op substations must have at least 2 busbars.

Because we don't want to have to make topological choices on how to deal when multiple buses are in the same substation 
in pypowybl we decided to double the busbars in the backend and to give the information to Grid2op as described in the doc :
[*This “coding” allows for easily mapping the bus id (each bus is represented with an id in pandapower) and whether its busbar 1 or busbar 2 (grid2op side). More precisely: busbar 1 of substation with id sub_id will have id sub_id and busbar 2 of the same substation will have id sub_id + n_sub (recall that n_sub is the number of substation on the grid).*](https://grid2op.readthedocs.io/en/latest/createbackend.html#:~:text=This%20%E2%80%9Ccoding%E2%80%9D%20allows,on%20the%20grid)

## Function part

### Usage in the code

### Functions for topological actions

#### Doubling buses (Pypowsybl backend)
This is done by calling the _double_buses function in the PowsyblBackend.py file. Like so the initial buses (in bus_breaker_view) are doubled but initially connected with none object. 

#### Usage of move_connectable (Pypowsybl backend)
This allows the user to move any object from one bus to another in the backend. This function could only use bus_breaker_view
buses id to work (different from bus_view buses id in pypowsybl). This is particularly useful for Grid2op topological 
changes to switch any object from a bus to another inside a substation (only possible topological action in Grid2op)