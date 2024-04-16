import itertools
import pickle
import os
import networkx as nx
from src.data_structures import Coords3d
from src.parameters import DEFAULT_VEHICLE_DENSITY, VEH_SPEED_RANGE, rng, VEH_ANTENNA_HEIGHT, SMALL_CYCLE_T, RANDOM_DENSITIES_RANGE
from typing import List
import numpy as np
import json
import heapq
import sympy
from pickle import HIGHEST_PROTOCOL
import multiprocessing
import bisect


JSON_NODES_PROBS_FILE = 'nodes_out.json'
JSON_ENTRY_DENSITIES_FILE = 'entry_densities.json'
JSON_NODE_DENSITIES_FILE = 'node_densities.json'

module_dir = os.path.dirname(os.path.abspath(__file__))

def get_bologna_streets():
    with open(module_dir + '\\bologna_graph.pkl', 'rb') as file:
        graph = pickle.load(file)
    return graph


class Street:
    def __init__(self, start_node_id: int, end_node_id: int, start_node_coords: Coords3d, end_node_coords: Coords3d):
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.start_node_coords = start_node_coords
        self.end_node_coords = end_node_coords
        self.length = end_node_coords.get_distance_to(start_node_coords, True)
        self.current_density = 0
        self.expected_density = 0
        self.expected_density_meter = 0
        t_end = self.length / VEH_SPEED_RANGE[0] + SMALL_CYCLE_T
        self.arrival_times = -1 * np.ones(int(1.2 * t_end * RANDOM_DENSITIES_RANGE[1]))
        self.fps = []
        self.vehicles = []
        self.one_arrived = False
        self.n_vs = 0
        self.dist_to_fp_start = None

    def generate_arrival_times(self, t_end=SMALL_CYCLE_T):
        self.arrival_times[:] = -1
        t_end = self.length / VEH_SPEED_RANGE[0] + t_end
        t, t_idx = 0, 0
        while t < t_end:
            assert t_idx < self.arrival_times.shape[0]
            interarrival_time = rng.exponential(
                1 / self.expected_density)  # Generate interarrival time from exponential distribution
            t += interarrival_time  # Update time and append arrival time to list
            self.arrival_times[t_idx] = t
            t_idx += 1
        self.arrival_t_idx = 0

    def generate_vehicle(self, arrival_time, small_t):
        speed = VEH_SPEED_RANGE[0]  # Min speed for now
        current_node_coords = self.start_node_coords.copy()
        dest_node_coords = self.end_node_coords.copy()
        assert dest_node_coords
        new_vehicle = Vehicle(current_node_coords.copy(), self.end_node_id, self.start_node_id, speed)
        new_vehicle.set_end_point(dest_node_coords.copy())
        if new_vehicle.update_coords(small_t - arrival_time):
            new_vehicle.in_simulation = False
        _v = new_vehicle
        return new_vehicle

    def update_vehicles_street(self, small_t, t_step):
        n_vs_satisfied_snr = 0
        while small_t >= self.arrival_times[self.arrival_t_idx]:
            new_t = self.arrival_times[self.arrival_t_idx]
            assert new_t > 0
            self.arrival_times[self.arrival_t_idx] = -1
            self.arrival_t_idx += 1
            self.vehicles.append(self.generate_vehicle(new_t, small_t))
            self.n_vs += 1

        for idx in reversed(range(len(self.vehicles))):
            vehicle = self.vehicles[idx]
            if not vehicle.update_coords(t_step):
                dist_from_start = vehicle.coords.get_distance_to(self.start_node_coords)
                vehicle.fp_idx = bisect.bisect_left(self.dist_to_fp_start, dist_from_start)
            if not vehicle.in_simulation:
                vehicle.in_simulation = False
                self.vehicles.pop(idx)
                self.one_arrived = True
                self.n_vs -= 1
                del vehicle


class StreetGraph:
    nodes = {}
    nodes_out = {}
    nodes_in = {}
    entry_nodes = []
    entry_nodes_densities = []
    small_t = 0
    large_t = 0
    small_t_end = SMALL_CYCLE_T
    streets = []
    node_densities = {}
    vehicles = property(fget=lambda self: [_v for street in self.streets for _v in street.vehicles])

    def __init__(self):
        self.graph = get_bologna_streets()
        self.extract_environment()

    def populate_streets_densities(self, load=True):
        for street in self.streets:
            street.expected_density = ((RANDOM_DENSITIES_RANGE[1] - RANDOM_DENSITIES_RANGE[0]) *
                                       rng.random() + RANDOM_DENSITIES_RANGE[0])
            street.expected_density_meter = street.expected_density/VEH_SPEED_RANGE[0]

    def extract_environment(self, load=True):
        self.populate_streets()
        self.populate_streets_densities(load=load)
        self.generate_vehicles_arrival_times(self.small_t_end)

    def generate_vehicles_arrival_times(self, t_end=SMALL_CYCLE_T):
        for street_idx, street in enumerate(self.streets):
            street.generate_arrival_times(t_end=t_end)

    def simulate_time_step(self, t_step):
        self.small_t += t_step
        self.large_t += t_step
        if self.small_t >= self.small_t_end:
            self.small_t = 0
            self.generate_vehicles_arrival_times(self.small_t_end)
        self.update_vehicles(t_step)

    def update_vehicles(self, t_step):
        for street_idx, street in enumerate(self.streets):
            street.update_vehicles_street(self.small_t, t_step=t_step)

        # args =[(_v, t_step) for _v in self.vehicles]
        # with multiprocessing.Pool(processes=6) as pool:
        #     new_vs_attr = pool.starmap(update_vehicle, args)
        #
        # for idx, vehicle in enumerate(self.vehicles):
        #     vehicle.coords.x = new_vs_attr[idx][0][0]
        #     vehicle.coords.y = new_vs_attr[idx][0][1]
        #     vehicle.in_simulation = new_vs_attr[idx][1]
        #
        # for idx, vehicle in enumerate(self.vehicles):
        #     update_vehicle(vehicle, t_step)

    def populate_streets(self):
        for start, end in self.graph.edges():
            start_pos = Coords3d(self.graph.nodes[start]['x'], self.graph.nodes[start]['y'], 0)
            end_pos = Coords3d(self.graph.nodes[end]['x'], self.graph.nodes[end]['y'], 0)
            self.streets.append(Street(start, end, Coords3d.from_array(start_pos), Coords3d.from_array(end_pos)))

    def update_street_densities(self):
        for idx, street in enumerate(self.streets):
            n_v = len(street.vehicles)
            street.current_density = n_v / street.length

    def save_data_to_share(self):
        _data = [self.nodes, self.nodes_out, self.nodes_in, self.entry_nodes, self.entry_nodes_densities]
        with open('street_objects.pkl', 'wb') as f:
            pickle.dump(_data, f, protocol=HIGHEST_PROTOCOL)
        with open('graph_object.pkl', 'wb') as f:
            pickle.dump(self.graph, f, protocol=HIGHEST_PROTOCOL)


def update_vehicle_queue(queue):
    while True:
        task = queue.get()
        if task is None:
            break
        update_vehicle(task[0], task[1])


# def update_vehicle(vehicle, t_step):
#     vehicle.update_coords(t_step)
#     return vehicle.coords.as_2d_array(), vehicle.in_simulation


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
        self.first_time = False
        self.fp_idx = None

    def update_coords(self, t_step):
        distance = t_step * self.speed + self.extra_dist
        self.extra_dist = 0
        reached_dest, self.extra_dist = self.coords.update(self.end_point, distance)
        if reached_dest:
            self.end_point = None
            self.in_simulation = False
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
    while 1:
        streets.simulate_time_step(1)
        print(streets.vehicles)
    # for _node in streets.nodes_out:
    #     sum = 0
    #     for edge in streets.nodes_out[_node]:
    #         sum += streets.nodes_out[_node][edge]['multinomial_prob']
    #     print(sum)
