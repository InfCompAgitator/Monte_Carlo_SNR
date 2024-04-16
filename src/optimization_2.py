from pulp import LpProblem, LpVariable, LpBinary, LpMinimize, lpSum, value
import pulp

from src.parameters import N_BEAMS_PER_DBS
from src.bologna.city import City, Footprint
import numpy as np
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    alpha = 0.7
    city = City()
    prob = LpProblem('DBS_Coverage', LpMinimize)
    num_dbs_locs = len(city.possible_locs)
    num_fp_locs = len(city.footprints_centers)
    w = np.zeros((num_dbs_locs, num_fp_locs))
    for i in range(num_dbs_locs):
        for j in range(num_fp_locs):
            dbs_loc_node_idx = i + num_fp_locs
            if city.visibility_graph.has_edge(dbs_loc_node_idx, j):
                w[i, j] = city.visibility_graph.get_edge_data(dbs_loc_node_idx, j).get('ap')
            else:
                w[i, j] = 0

    w_binary = w > 0

    # Decision variables
    x = LpVariable.dicts('x', ((i, j) for i in range(num_dbs_locs) for j in range(num_fp_locs)), cat='Binary')
    y = LpVariable.dicts('y', (i for i in range(num_dbs_locs)), cat='Binary')

    # Objective function
    prob += (1 - alpha) * lpSum(y[i] for i in range(num_dbs_locs)) + alpha * pulp.lpSum(x[i, j] * w[i, j] for i in range(num_dbs_locs) for j in range(num_fp_locs))

    # Constraints
    for j in range(num_fp_locs):
        prob += lpSum(x[i, j] for i in range(num_dbs_locs)) == 1

    for i in range(num_dbs_locs):
        prob += lpSum(x[i, j] for j in range(num_fp_locs)) <= N_BEAMS_PER_DBS * y[i]

    for i in range(num_dbs_locs):
        prob += lpSum(x[i, j] for j in range(num_fp_locs)) >= y[i]

    for i in range(num_dbs_locs):
        for j in range(num_fp_locs):
            prob += x[i, j] <= w_binary[i, j]
    print("Solving Problem!")
    # prob.solve(pulp.CPLEX_CMD(msg=True, path=r'C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2211\cplex\bin\x64_win64\cplex.exe'))
    prob.solve(pulp.CPLEX_CMD(msg=True, path=r'C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe'))

    x_sol = np.zeros((num_dbs_locs, num_fp_locs))
    for i in range(num_dbs_locs):
        for j in range(num_fp_locs):
            x_sol[i,j] = pulp.value(x[i, j])


if __name__ == '__main__':
    city = City()
    prob = LpProblem('DBS_Coverage', LpMinimize)
    num_dbs_locs = len(city.possible_locs)
    num_fp_locs = len(city.footprints_centers)
    w = np.zeros((num_dbs_locs, num_fp_locs))
    for i in range(num_dbs_locs):
        for j in range(num_fp_locs):
            dbs_loc_node_idx = i + num_fp_locs
            if city.visibility_graph.has_edge(dbs_loc_node_idx, j):
                w[i, j] = city.visibility_graph.get_edge_data(dbs_loc_node_idx, j).get('ap')
            else:
                w[i, j] = 0

    w_binary = w > 0

    # Decision variables
    x = LpVariable.dicts('x', ((i, j) for i in range(num_dbs_locs) for j in range(num_fp_locs)), cat='Binary')
    y = LpVariable.dicts('y', (i for i in range(num_dbs_locs)), cat='Binary')

    # Objective function
    prob += lpSum(y[i] for i in range(num_dbs_locs))

    # Constraints
    for j in range(num_fp_locs):
        prob += lpSum(x[i, j] for i in range(num_dbs_locs)) == 1

    for i in range(num_dbs_locs):
        prob += lpSum(x[i, j] for j in range(num_fp_locs)) <= N_BEAMS_PER_DBS * y[i]

    for i in range(num_dbs_locs):
        prob += lpSum(x[i, j] for j in range(num_fp_locs)) >= y[i]

    for i in range(num_dbs_locs):
        for j in range(num_fp_locs):
            prob += x[i, j] <= w_binary[i, j]
    print("Solving Problem!")
    # prob.solve(pulp.CPLEX_CMD(msg=True, path=r'C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2211\cplex\bin\x64_win64\cplex.exe'))
    prob.solve(pulp.CPLEX_CMD(msg=True, path=r'C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe'))

    x_sol = np.zeros((num_dbs_locs, num_fp_locs))
    for i in range(num_dbs_locs):
        for j in range(num_fp_locs):
            x_sol[i,j] = pulp.value(x[i, j])
