import pickle
import os
import networkx as nx
from src.data_structures import Coords3d
from src.parameters import DEFAULT_VEHICLE_DENSITY, VEH_SPEED_RANGE, rng, VEH_ANTENNA_HEIGHT, SMALL_CYCLE_T
from typing import List
import numpy as np
import json
import heapq
import sympy

JSON_NODES_PROBS_FILE = 'nodes_out.json'
JSON_ENTRY_DENSITIES_FILE = 'entry_densities.json'
JSON_NODE_DENSITIES_FILE = 'node_densities.json'
RANDOM_DENSITIES_RANGE = [1, 4]


def get_bologna_streets():
    with open(os.getcwd() + '\\bologna_graph.pkl', 'rb') as file:
        graph = pickle.load(file)
    return graph


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
def get_linearly_dependent_rows_2(matrix):
    dep_rows = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i != j:
                inner_product = np.inner(
                    matrix[:,i],
                    matrix[:,j]
                )
                norm_i = np.linalg.norm(matrix[:,i])
                norm_j = np.linalg.norm(matrix[:,j])

                if np.abs(inner_product - norm_j * norm_i) < 1E-15:
                    dep_rows.append(j)
    return dep_rows
def get_linearly_dependent_rows(mat):
    _, inds = sympy.Matrix(mat).T.rref()
    return set(range(mat.shape[0])).difference(set(inds))


# class Street:
#     density = 0
#     f_streets = []
#
#     def __init__(self, start_coords: Coords3d, end_coords: Coords3d, f_streets=[], f_streets_mn_probs=[],
#                  v_density=DEFAULT_VEHICLE_DENSITY, street_id: int = None):
#         """
#         :param start_coords: coordinates of the street start point.
#         :param end_coords: coordinates of the street end.
#         :param f_streets: feeding streets. I.e., list of streets with cars leaving them and possibly entering this street.
#         :param f_streets_mn_probs: at index i, the probability of a car from street f_streets[i], entering this street.
#          """
#
#         self.id = next(self._ids) if street_id is None else street_id
#         self.start_coords = start_coords
#         self.end_coords = end_coords
#         self.f_streets = f_streets
#         if not len(f_streets):
#             self.density = v_density
#             self.is_entry = True
#         else:
#             for idx, _street in enumerate(f_streets):
#                 self.density += f_streets_mn_probs[idx] * _street.density
#
#     def __eq__(self, other):
#         if isinstance(other, Street):
#             return self.id == other.id
#         return NotImplemented

class Street:
    def __init__(self, start_node_id: int, end_node_id: int, start_node_coords: Coords3d, end_node_coords: Coords3d):
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.start_node_coords = start_node_coords
        self.end_node_coords = end_node_coords
        self.length = end_node_coords.get_distance_to(start_node_coords, True)
        self.current_density = 0
        self.expected_density = 0
        self.fps = []


class StreetGraph:
    nodes = {}
    nodes_out = {}
    nodes_in = {}
    entry_nodes = []
    entry_nodes_densities = []
    arrival_times = []
    small_t = 0
    large_t = 0
    small_t_end = SMALL_CYCLE_T
    vehicles = []
    streets = []
    node_densities = {}

    def __init__(self):
        self.graph = get_bologna_streets()
        self.extract_environment()

    def populate_entry_densities(self, load=True):
        if not load:
            self.entry_nodes_densities = [DEFAULT_VEHICLE_DENSITY for _ in range(len(self.entry_nodes))]
        else:
            with open(JSON_ENTRY_DENSITIES_FILE, 'r') as f:
                self.entry_nodes_densities = json.load(f)

    def extract_environment(self, load=True):
        def extract_node_rec(node, nodes_dict, nodes_out_dict, nodes_in_dict, load=load):
            if load:
                with open(JSON_NODES_PROBS_FILE, 'r') as f:
                    _old_data = json.load(f)
            if node in nodes_out_dict:
                return
            _successors = list(self.graph.successors(node))

            if not load:
                if len(_successors) > 1:
                    multinomial_probs = rng.dirichlet(np.ones(len(_successors)), size=1)[0].round(4)
                    while multinomial_probs.sum() != 1:
                        multinomial_probs = rng.dirichlet(np.ones(len(_successors)), size=1)[0].round(4)
                else:
                    multinomial_probs = np.array([1], dtype=np.float64)
                # _ = rng.random(len(_successors))
                # multinomial_probs = (_ / _.sum()
                if len(multinomial_probs) > 0:
                    assert (multinomial_probs.sum() == 1)
            elif len(_successors) > 0:
                multinomial_probs = [_old_data[f'{node}'][i]['multinomial_prob'] for i in _old_data[f'{node}']]
            else:
                multinomial_probs = []

            for _successor, _prob in zip(_successors, multinomial_probs):
                # if _successor in nodes_in_dict and node in nodes_out_dict:
                #     if node in nodes_in_dict[_successor] and _successor in nodes_out_dict[node]:
                #         continue
                if node not in nodes_dict:
                    nodes_dict[node] = {}
                if node not in nodes_out_dict:
                    nodes_out_dict[node] = {}
                if _successor not in nodes_in_dict:
                    nodes_in_dict[_successor] = {}

                nodes_dict[node][_successor] = {'multinomial_prob': _prob}
                nodes_out_dict[node][_successor] = {'multinomial_prob': _prob}
                nodes_in_dict[_successor][node] = {'multinomial_prob': _prob}

                current_node = _successor

                _nodes_dict = nodes_dict[node]
                _nodes_out_dict = nodes_out_dict
                _nodes_in_dict = nodes_in_dict

                extract_node_rec(current_node, _nodes_dict, _nodes_out_dict, _nodes_in_dict)

        self.entry_nodes = [node for node, attrs in self.graph.nodes(data=True) if attrs.get('entry') == True]

        for _entry_node in self.entry_nodes:
            self.nodes_in[_entry_node] = {}
            _node = _entry_node
            extract_node_rec(_node, self.nodes, self.nodes_out, self.nodes_in)
        self.populate_entry_densities(load=load)
        self.generate_vehicles_arrival_times(self.small_t_end)
        self.populate_streets()
        self.calculate_expected_densities()

    def generate_vehicles_arrival_times(self, T_end=SMALL_CYCLE_T):
        self.arrival_times = [[] for _ in range(len(self.entry_nodes))]
        self.small_t_end = T_end
        arrival_t_end = T_end
        for node_idx, node in enumerate(self.entry_nodes):
            t = 0
            while t < arrival_t_end + 10:
                interarrival_time = rng.exponential(
                    1 / self.entry_nodes_densities[node_idx])
                t += interarrival_time
                self.arrival_times[node_idx].append(t)
        return self.arrival_times

    def simulate_time_step(self, t_step):
        self.small_t += t_step
        self.large_t += t_step
        if self.small_t >= self.small_t_end:
            self.small_t = 0
            self.generate_vehicles_arrival_times(self.small_t_end)
        self.update_vehicles(t_step)

    def update_vehicles(self, t_step):
        for node_idx, node in enumerate(self.entry_nodes):
            while self.small_t >= self.arrival_times[node_idx][0]:
                new_t = heapq.heappop(self.arrival_times[node_idx])
                multinomial_probs = [self.nodes_out[node][i]['multinomial_prob'] for i in
                                     self.nodes_out[node]]
                selected_idx = rng.choice(range(len(multinomial_probs)), p=multinomial_probs)
                dest_node = list(self.nodes_out[node])[selected_idx]
                self.generate_vehicle(new_t, dest_node, node)
        for idx, vehicle in enumerate(self.vehicles):
            if vehicle.update_coords(t_step):
                if not self.update_vehicle_endpoint(vehicle):
                    self.vehicles.pop(idx)
                    del vehicle

    def generate_vehicle(self, arrival_time, dest_node, current_node):
        speed = VEH_SPEED_RANGE[0]  # Min speed for now
        current_node_coords = Coords3d(self.graph.nodes[current_node]['x'], self.graph.nodes[current_node]['y'],
                                       VEH_ANTENNA_HEIGHT[0])
        dest_node_coords = Coords3d(self.graph.nodes[dest_node]['x'], self.graph.nodes[dest_node]['y'],
                                    VEH_ANTENNA_HEIGHT[0])
        new_vehicle = Vehicle(current_node_coords.copy(), dest_node, current_node, speed)
        new_vehicle.set_end_point(dest_node_coords.copy())
        if new_vehicle.update_coords(self.small_t - arrival_time):
            self.update_vehicle_endpoint(new_vehicle)
        _v = new_vehicle
        self.vehicles.append(new_vehicle)

    def update_vehicle_endpoint(self, vehicle):
        vehicle.current_node = vehicle.dest_node
        if vehicle.dest_node not in self.nodes_out:
            vehicle.in_simulation = False
            return False
        multinomial_probs = [self.nodes_out[vehicle.dest_node][i]['multinomial_prob'] for i in
                             self.nodes_out[vehicle.dest_node]]
        selected_idx = rng.choice(range(len(multinomial_probs)), p=multinomial_probs)
        vehicle.dest_node = list(self.nodes_out[vehicle.dest_node])[selected_idx]
        vehicle.set_end_point(Coords3d(self.graph.nodes[vehicle.dest_node]['x'],
                                       self.graph.nodes[vehicle.dest_node]['y'],
                                       VEH_ANTENNA_HEIGHT[0]))
        return True

    def populate_streets(self):
        for start, end in self.graph.edges():
            start_pos = Coords3d(self.graph.nodes[start]['x'], self.graph.nodes[start]['y'], 0)
            end_pos = Coords3d(self.graph.nodes[end]['x'], self.graph.nodes[end]['y'], 0)
            self.streets.append(Street(start, end, Coords3d.from_array(start_pos), Coords3d.from_array(end_pos)))

    def check_all_have_path(self):
        for node in self.graph.nodes:
            flag = False
            for node_2 in self.graph.nodes:
                if nx.has_path(self.graph, node_2, node):
                    flag = True
            if not flag:
                raise Exception('No Path!')
        # for node in self.graph.nodes:
        #     flag = False
        #     for _entry in self.entry_nodes:
        #         if nx.has_path(self.graph, _entry, node):
        #             flag = True
        #     if not flag:
        #         raise Exception('No Path!')

    def update_street_densities(self):
        for idx, street in enumerate(self.streets):
            n_v = 0
            for vehicle in self.vehicles:
                if vehicle.current_node == street.start_node_id and vehicle.dest_node == street.end_node_id:
                    n_v += 1
            street.current_density = n_v / street.length

    def calculate_expected_densities(self):
        # Random for now
        for node in self.graph.nodes:
            if node not in self.entry_nodes:
                self.node_densities[node] = ((RANDOM_DENSITIES_RANGE[1] - RANDOM_DENSITIES_RANGE[0]) *
                                             rng.random() + RANDOM_DENSITIES_RANGE[0])
            else:
                self.node_densities[node] = self.entry_nodes_densities[self.entry_nodes.index(node)]

        for street in self.streets:
            street.expected_density = self.node_densities[street.start_node_id] * \
                                      self.nodes_in[street.end_node_id][street.start_node_id]['multinomial_prob']

        A = np.zeros((len(self.graph.nodes), len(self.graph.nodes)))
        B = np.zeros((len(self.graph.nodes)))
        for node_idx, node in enumerate(self.graph.nodes):
            if node in self.entry_nodes:
                B[node_idx] = self.entry_nodes_densities[self.entry_nodes.index(node)]
                A[node_idx][node_idx] = 1
                continue
            else:
                A[node_idx][node_idx] = -1
            for node_idx_2, node_2 in enumerate(self.graph.nodes):
                if node_2 in self.nodes_in[node]:
                    A[node_idx][node_idx_2] = self.nodes_in[node][node_2]['multinomial_prob']


        A = np.zeros((len(self.graph.nodes) + 2, len(self.graph.nodes) + 2))
        B = np.zeros((len(self.graph.nodes) + 2))
        for node_idx, node in enumerate(self.graph.nodes):
            if node not in self.nodes_out:
                continue
            for node_idx_2, node_2 in enumerate(self.graph.nodes):
                if node_2 in self.nodes_out[node]:
                    A[node_idx][node_idx_2] = self.nodes_out[node][node_2]['multinomial_prob']
        #Sink state
        node_idx += 1
        for node_idx_2, node_2 in enumerate(self.graph.nodes):
            if node_2 not in self.nodes_out or self.nodes_out[node_2] == {}:
                A[node_idx_2][node_idx] = 1

        A[node_idx][node_idx + 1] = 1
        # Source state
        node_idx += 1
        for node_idx_2, node_2 in enumerate(self.graph.nodes):
            if node_2 in self.entry_nodes:
                A[node_idx][node_idx_2] = self.entry_nodes_densities[self.entry_nodes.index(node_2)]/np.sum(self.entry_nodes_densities)
        B[node_idx] = np.sum(self.entry_nodes_densities)
        densities = np.linalg.solve(A, B)

        def steady_state_prop(p):
            dim = p.shape[0]
            q = (p - np.eye(dim))
            ones = np.ones(dim)
            q = np.c_[q, ones]
            QTQ = np.dot(q, q.T)
            bQT = np.ones(dim)
            return np.linalg.solve(QTQ, bQT)


class Vehicle:
    def __init__(self, coords: Coords3d, dest_node: int, init_node: int, speed=VEH_SPEED_RANGE[0]):
        self.coords = coords.copy()
        self.transceiver = None
        self.speed = speed
        self.in_simulation = True
        self.color = 'b'
        self.coords.z = 0
        # self.coords.z = rng.uniform(VEH_ANTENNA_HEIGHT[0], VEH_ANTENNA_HEIGHT[1])
        self.lane_direction = None
        self.end_point = None
        self.extra_dist = 0
        self.current_node = init_node
        self.dest_node = dest_node

    def update_coords(self, t_step):
        distance = t_step * self.speed + self.extra_dist
        self.extra_dist = 0
        reached_dest, self.extra_dist = self.coords.update(self.end_point, distance)
        if reached_dest:
            self.end_point = None
            return True
        else:
            return False

    def set_end_point(self, end_point: Coords3d):
        self.end_point = end_point


if __name__ == '__main__':
    streets = StreetGraph()
    # streets.check_all_have_path()
    # streets.generate_vehicles_arrival_times()
    # with open(JSON_ENTRY_DENSITIES_FILE, 'w') as f:
    #     json.dump(streets.entry_nodes_densities, f)
    #
    # with open(JSON_NODES_PROBS_FILE, 'w') as f:
    #     json.dump(streets.nodes_out, f)
    # while 1:
    #     streets.simulate_time_step(20)
    # for _node in streets.nodes_out:
    #     sum = 0
    #     for edge in streets.nodes_out[_node]:
    #         sum += streets.nodes_out[_node][edge]['multinomial_prob']
    #     print(sum)
