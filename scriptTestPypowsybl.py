import pypowsybl.network as pp
import pypowsybl.loadflow as lf
import pypowsybl as ppow
import pandapower as pdp


def _double_buses(grid):

    df = grid.get_buses()
    L = []
    for elem in grid.get_voltage_levels().index:
        for bus_id in grid.get_bus_breaker_topology(voltage_level_id=elem).buses.index:
            L.append(bus_id)
    L_voltage_id = df['voltage_level_id'].to_list()
    for i in range(len(L)):
        grid.create_buses(id=L[i] + '_bis', voltage_level_id=L_voltage_id[i], name=df['name'][i])

if  __name__ == "__main__":
    network_pp = pp.create_ieee14()
    _double_buses(network_pp)
    ppow.network.move_connectable(network=network_pp, equipment_id='B1-G', bus_origin_id='B1',
                                  bus_destination_id='B1_bis')
    print(network_pp.get_generators(all_attributes=True))
    results_pypow = lf.run_ac(network_pp)
