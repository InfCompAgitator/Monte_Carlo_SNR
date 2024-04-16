from matplotlib import pyplot as plt

from src.bologna.obstacles import get_bologna_buildings
from src.bologna.streets_no_markov import StreetGraph
import numpy as np
from src.data_structures import Coords3d
from src.parameters import VEH_ANTENNA_HEIGHT, DBS_HEIGHT, LOCS_DENSITIES_PER_STREET_LENGTH, BEAM_LENGTH, STREET_WIDTH
from matplotlib.animation import FuncAnimation
import matplotlib;
from math import copysign
import networkx as nx
from src.visibility_tools.visibility_polygon_ctrl import get_vis_polygon, get_vertices_in_polygon
from itertools import count
import pickle
import tqdm
from src.channel_modelling.channel_model_beamforming import calculate_beam_access_probability
import time
import os

# matplotlib.use("TkAgg")
VIS_GRAPH_FILE = 'vis_graph.pkl'
fig, ax = plt.subplots()
vehs_plot = []

module_dir = os.path.dirname(os.path.abspath(__file__))

class Footprint:
    _ids = count(0)

    def __init__(self, center_coords: Coords3d, id=None, street_id=None, start_coords=None, end_coords=None):
        self.center_coords = center_coords
        self.id = next(self._ids) if id is None else id
        self.street_id = street_id
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.side_coords = None
        self.bf_gain = None
        self.serving_dbs_loc = None


class City:
    plot_flag = False
    ani = None
    possible_locs = []
    footprints_centers = []

    def __init__(self):
        self.visibility_graph = None
        self.buildings = get_bologna_buildings()
        self.segments = self.get_obstacles_segments()
        self.street_graph = StreetGraph()
        self.buildings_segments = self.get_obstacles_segments()
        self.set_possible_dbs_coords()
        self.set_footprints_coords()
        self.build_visibility_graph()

    def simulate_time_step(self, t_step=1):
        self.street_graph.simulate_time_step(t_step)
        if self.plot_flag:
            fig.canvas.flush_events()
            plt.pause(1)

    def init_plot(self):
        self.plot_buildings()
        self.plot_streets(False, fig, ax)
        ax.set_aspect('equal')
        return vehs_plot

    def update_plot(self, i):
        for _v in vehs_plot[:]:
            _v.remove()
        vehs_plot.clear()
        new_coords = np.array([v.coords.as_2d_array() for v in self.street_graph.vehicles if v.in_simulation])
        colors = [v.color for v in self.street_graph.vehicles if v.in_simulation]
        for (x, y), _c in zip(new_coords, colors):
            veh_plot, = ax.plot(x, y, color=_c, marker='x', markersize=3, alpha=0.5)
            vehs_plot.append(veh_plot)
        return vehs_plot

    def generate_plot(self):
        self.ani = FuncAnimation(fig, self.update_plot,
                                 init_func=self.init_plot, blit=False, interval=1000, frames=None, repeat=False)
        self.plot_flag = True
        plt.ion()
        plt.show()

    def get_obstacles_segments(self, flatten=True):
        segs = self.buildings.get_total_segments()[0]
        return segs if not flatten else [item for sublist in segs for item in sublist]

    def plot_buildings(self, show_flag=False):
        _rects = self.buildings.plot_obstacles(False, 'gray', True)
        bds = self.buildings.get_margin_boundary(False)
        for _rect in _rects:
            ax.add_patch(_rect)
        ax.set_ylim([bds[1][0], bds[1][1]])
        ax.set_xlim([bds[0][0], bds[0][1]])
        if show_flag:
            fig.show()
        return fig, ax

    def plot_streets(self, show_flag=False, fig=None, ax=None):
        if fig is None:
            fig, ax = plt.subplots()
        for start, end in self.street_graph.graph.edges():
            start_pos = (self.street_graph.graph.nodes[start]['x'], self.street_graph.graph.nodes[start]['y'])
            end_pos = (self.street_graph.graph.nodes[end]['x'], self.street_graph.graph.nodes[end]['y'])

            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-', alpha=0.5)  # 'k-' for black line
            # ax.text((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2, f'{start} -> {end}', fontsize=4,
            #         rotation= np.arctan((start_pos[1] - end_pos[1])/ (start_pos[0] - end_pos[0])) * 180/np.pi, ha='center', va='center')
        for _street in self.street_graph.streets:
            ax.plot([_street.start_node_coords.x, _street.end_node_coords.x],
                    [_street.start_node_coords.y, _street.end_node_coords.y], 'r-', alpha=0.5)
        ax.set_aspect('equal')
        # plt.xlim(min(self.street_graph.graph.nodes[node]['x'] for node in self.street_graph.graph.nodes()) - 0.01,
        #          max(self.street_graph.graph.nodes[node]['x'] for node in self.street_graph.graph.nodes()) + 0.01)
        # plt.ylim(min(self.street_graph.graph.nodes[node]['y'] for node in self.street_graph.graph.nodes()) - 0.01,
        #          max(self.street_graph.graph.nodes[node]['y'] for node in self.street_graph.graph.nodes()) + 0.01)
        if show_flag:
            fig.show()
        return fig, ax

    def plot_footprints(self, fig, ax):
        for fp in self.footprints_centers:
            ax.plot(fp.center_coords.x, fp.center_coords.y, marker='o', markersize=3, markerfacecolor='none',
                    markeredgecolor='g')
            # ax.plot(fp.side_coords.x, fp.side_coords.y, marker='.', markersize=1, markerfacecolor='none', markeredgecolor='k')

    def plot_dbs_locs(self, fig, ax):
        for loc in self.possible_locs:
            ax.plot(loc.x, loc.y, marker='x', markersize=3, markerfacecolor='none', markeredgecolor='b')
            # ax.plot(fp.side_coords.x, fp.side_coords.y, marker='.', markersize=1, markerfacecolor='none', markeredgecolor='k')

    def plot_visibility(self, fig, ax):
        for _edge in self.visibility_graph.edges():
            dbs_loc = self.possible_locs[_edge[0] - len(self.footprints_centers)]
            fp_loc = self.footprints_centers[_edge[1]].center_coords
            ax.plot([dbs_loc.x, fp_loc.x], [dbs_loc.y, fp_loc.y], 'k-', linewidth=1, alpha=0.5)

    def set_possible_dbs_coords(self):
        # TODO: Harder SAVE TO FILE
        # Nodes coords
        for _node in self.street_graph.graph.nodes(data=True):
            self.possible_locs.append(Coords3d(_node[1]['x'], _node[1]['y'],
                                               DBS_HEIGHT))
        # Streets Divisions
        for street in self.street_graph.streets:
            n_points = np.ceil(street.length * LOCS_DENSITIES_PER_STREET_LENGTH).astype(int)
            xs, ys = np.linspace(
                (street.start_node_coords.x, street.start_node_coords.y),
                (street.end_node_coords.x, street.end_node_coords.y), num=n_points, endpoint=False).T
            for x, y in zip(xs, ys):
                self.possible_locs.append(Coords3d(x, y, DBS_HEIGHT))

    def set_footprints_coords(self):
        # TODO:Harder Save to file
        for idx, street in enumerate(self.street_graph.streets):
            n_foot_prints = int(np.ceil(street.length / BEAM_LENGTH))
            del_x_y = street.end_node_coords - street.start_node_coords
            normal_vector = Coords3d(del_x_y.y, -del_x_y.x, 0)
            normal_vector = normal_vector / normal_vector.norm()
            street_direction = del_x_y.y / del_x_y.x
            d_x_1 = np.sqrt(((BEAM_LENGTH / 2) ** 2) / (1 + street_direction ** 2)) * copysign(1, del_x_y.x)
            d_y_1 = street_direction * d_x_1
            start_coords_x = street.start_node_coords.x + d_x_1
            start_coords_y = street.start_node_coords.y + d_y_1
            d_x_2 = np.sqrt(((BEAM_LENGTH) ** 2) / (1 + street_direction ** 2)) * copysign(1, del_x_y.x)
            d_y_2 = street_direction * d_x_2
            current_coord = Coords3d(start_coords_x, start_coords_y, 0)

            if (current_coord - street.start_node_coords).norm() < street.length:
                new_fp = Footprint(Coords3d(start_coords_x, start_coords_y, 0), street_id=idx,
                                   start_coords=Coords3d(start_coords_x - d_x_1, start_coords_y - d_y_1, 0),
                                   end_coords=Coords3d(start_coords_x + d_x_1, start_coords_y + d_y_1, 0))
                new_fp.side_coords = normal_vector * STREET_WIDTH + new_fp.center_coords
                street.fps.append(new_fp)
                self.footprints_centers.append(new_fp)
            else:
                new_fp = Footprint((street.start_node_coords + street.end_node_coords) / 2, street_id=idx,
                                   start_coords=street.start_node_coords, end_coords=street.end_node_coords)
                new_fp.side_coords = normal_vector * STREET_WIDTH + new_fp.center_coords
                street.fps.append(new_fp)
                self.footprints_centers.append(new_fp)
            prev_fp = new_fp
            for i in range(n_foot_prints - 1):
                current_coord.x += d_x_2
                current_coord.y += d_y_2
                if (current_coord - street.start_node_coords).norm() < street.length:
                    new_fp = Footprint(current_coord.copy(), street_id=idx,
                                       start_coords=current_coord - Coords3d(d_x_1, d_y_1, 0),
                                       end_coords=current_coord + Coords3d(d_x_1, d_y_1, 0))
                    new_fp.side_coords = normal_vector * STREET_WIDTH + new_fp.center_coords
                    self.footprints_centers.append(new_fp)
                    street.fps.append(new_fp)
                else:
                    new_fp = Footprint((prev_fp.center_coords + street.end_node_coords) / 2, street_id=idx,
                                       start_coords=prev_fp.center_coords, end_coords=street.end_node_coords)
                    new_fp.side_coords = normal_vector * STREET_WIDTH + new_fp.center_coords
                    self.footprints_centers.append(new_fp)
                    street.fps.append(new_fp)
                prev_fp = new_fp

        for idx, street in enumerate(self.street_graph.streets):
            n_fps = len(street.fps)
            street.dist_to_fp_start = np.zeros(n_fps)
            for fp_idx, _fp in enumerate(street.fps):
                street.dist_to_fp_start[idx] = _fp.start_coords.get_distance_to(street.start_node_coords)


    def build_visibility_graph(self, save=False, load=True):
        # TODO: Harder load from file
        if not load:
            self.visibility_graph = nx.Graph()
            n_footprints = next(Footprint._ids)
            [self.visibility_graph.add_node(loc_idx + n_footprints) for loc_idx in range(len(self.possible_locs))]
            [self.visibility_graph.add_node(fp_idx) for fp_idx in range(len(self.footprints_centers))]
            fps_2d = [_fp.center_coords.as_2d_array() for _fp in self.footprints_centers]
            for loc_idx, dbs_loc in tqdm.tqdm(enumerate(self.possible_locs), total=len(self.possible_locs)):
                vis_poly = get_vis_polygon(self.segments, dbs_loc.x, dbs_loc.y)
                v_pts = get_vertices_in_polygon(vis_poly, fps_2d)
                idxs = np.argwhere(v_pts[1]).flatten()
                for fp_idx in idxs:
                    dbs_loc_idx = loc_idx + n_footprints
                    access_prob, bf_gain = self.calculate_beam_access_probability(fp_idx, loc_idx)
                    if access_prob > 0:
                        self.visibility_graph.add_edge(dbs_loc_idx, fp_idx, fp_obj=self.footprints_centers[fp_idx], ap= access_prob, bf_gain=bf_gain)
        else:
            with open(module_dir + '\\' + VIS_GRAPH_FILE, 'rb') as file:
                self.visibility_graph = pickle.load(file)
        if save:
            with open(module_dir + '\\' + VIS_GRAPH_FILE, 'wb') as _file:
                pickle.dump(self.visibility_graph, _file)

    def calculate_beam_access_probability(self, fp_idx=200, dbs_loc_idx=565):
        # list(self.visibility_graph.edges())[0]
        street_id = self.footprints_centers[fp_idx].street_id
        street_density = self.street_graph.streets[street_id].expected_density_meter
        fp = self.footprints_centers[fp_idx]
        dbs_coords = self.possible_locs[dbs_loc_idx]
        ap_old, bf_gain, _, _, _, _, ap_n_1, ap_n_2 = calculate_beam_access_probability(street_density, fp, dbs_coords,
                                                                                  BEAM_LENGTH, STREET_WIDTH)
        return ap_n_2, bf_gain


if __name__ == '__main__':
    bologna = City()
    bologna.calculate_beam_access_probability()
    # bologna.generate_plot()
    idx = 0
    t_1 = time.perf_counter()
    densities = np.zeros_like(bologna.street_graph.streets)
    while 1:
        idx += 1
        bologna.simulate_time_step(1)
        expected_densities = np.array([street.expected_density_meter for street in bologna.street_graph.streets])
        densities = densities + (np.array([street.current_density for street in bologna.street_graph.streets]) -densities )/idx
        if idx % 20 == 0:
            bologna.street_graph.update_street_densities()
            print("Iteration:", idx)
            print([(street.current_density, street.expected_density_meter) for street in bologna.street_graph.streets])
            print([(densities[idx], street.expected_density_meter) for idx, street in enumerate(bologna.street_graph.streets)])
            print("MSE: ", np.mean((expected_densities - densities)**2))



    #     if not idx%20:
    #         bologna.street_graph.update_street_densities()
    #         print([street.current_density for street in bologna.street_graph.streets])
    # fig, ax = bologna.plot_buildings()
    # bologna.plot_streets(False, fig, ax)
    # # bologna.plot_footprints(fig, ax)
    # bologna.plot_dbs_locs(fig, ax)
    # bologna.plot_visibility(fig, ax)
    # fig.show()
