# pypowsybl-grid2opbackend integration (AIRGo project)

### Attribution
*This library is part of a project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101016508.*

![AIRGoLogo](logos/i-nergy_logo.png)

## Prerequisite
To be able to work properly with this backend integration you will have to install a specific version of pypowsybl.
You can find it there : [specific pypowsybl branch](https://github.com/powsybl/pypowsybl/tree/move_connectable_prototype).

### Installation process
You will have to follow the requirements from the pypowsybl repo [build from sources](https://github.com/powsybl/pypowsybl#build-from-sources) 
I copy/paste here a version of those requirements (from 10/10/2023).

> ## Build from sources
>
>That section is intended for developers who wish to build pypowsybl from the sources in this repository.
>
>Requirements:
>
>- Maven >= 3.1
>- Cmake >= 3.14
>- C++11 compiler
>- Python >= 3.7 for Linux, Windows and MacOS amd64
>- Python >= 3.8 for MacOS arm64
>- [Oracle GraalVM Java 17](https://www.graalvm.org/downloads/)
>
>To build from sources and install PyPowSyBl package:
>
>```bash
>git clone --recursive https://github.com/powsybl/pypowsybl.git
>export JAVA_HOME=<path to GraalVM>
>pip install --upgrade setuptools pip
>pip install -r requirements.txt
>pip install .
>```
>
>While developing, you may find it convenient to use the developer (or editable)
>mode of installation:
>
>```bash
>pip install -e .
># or, to build the C extension with debug symbols:
>python setup.py build --debug develop --user
>```
>
>Please refer to pip and setuptools documentations for more information.
>
>To run unit tests:
>
>```bash
>pytest tests
>```
>
>To run static type checking with `mypy`:
>```bash
>mypy -p pypowsybl
>```
>
>To run linting inspection with `pylint`:
>```bash
>pylint pypowsybl
>```
>
## Simple example of use
In the script [ScriptForSimpleUseCase.py](pypowsybl_grid2opbackend/ScriptForSimpleUseCase.py) you can find an example of simple agent doing 
one action using our backend on the ieee14 case network. Some several actions could be taken up for you to 
comment/decomment to act as you like on the network.

## AirGo project dataset generation
The dataset created for the needs of the project can be found [here](https://www.ai4europe.eu/research/ai-catalog/airgo-i-nergy-open-dataset).

### Processes of creation
This dataset was created using [chronics_creator.py](chronics_chreator.py). We generated data for january for exactly 4 
weeks of 7 days with a 5 minutes step resolution. The given dates do not correspond to any real date. Every month starting 
with a monday.

For some simplification purpose each week of a given month have exactly the same probabilistic distribution, what differentiates  
them is only the randomisation seed chose.

### Potential use for machine learning
In a classical machine learning training example we could decide for example to choose 3 weeks for training and 1 week for 
evaluation purposes. Nevertheless, it is also possible to separate more precisely data based on time step for example, but 
it will be more complicated.

### How to create some more synthetic data
If the available data are not enough we can create some more by changing the *nb_of_week* parameter in the main of the file.
The data will be created starting from the month of january until december, 4 weeks per month each.

## Definition and comparison of objects between Pypowsybl and Grid2op

### Lines and transformers

#### Pypowsybl
In Pypowsybl there is a real distinction between lines, 2 winding transformers and 3 winding transformers.

#### Grid2op
In Grid2op all those objects are assimilated as lines. Nevertheless, Grid2op knows which of these lines are real powerlines 
and those that are not.

#### Use of pandapower format to test pypowsybl backend
Because of the converter chain (Pandapower format -> Matpower -> Pypowsybl inner format), issues often happen and some
lines or transfos are considered by pypowsybl as the opposite. A way to see those changes is to analyze the lines
that have a null resistance (they were transfos in pandapower format), but it is still a workaround and not a solution
or complete analysis.


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
We chose to have 2 busbars at each substation to facilitate the integration of our backend with existent tests but this 
remains a personal choice, the explanation is bellow.

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

## License information
Copyright 2023 Artelys: http://www.artelys.com

This Source Code is subject to the terms of the Mozilla Public License (MPL) v2 also available
[here](https://www.mozilla.org/en-US/MPL/2.0/) or [in the repository](LICENSE).