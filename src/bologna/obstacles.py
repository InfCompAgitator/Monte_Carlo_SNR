#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from src.data_structures import Obstacle
from src.math_tools import line_point_angle, get_mid_azimuth
from itertools import chain as iter_chain
from src.parameters import BOUNDARY_MARGIN

module_dir = os.path.dirname(os.path.abspath(__file__))

class Obstacles(object):
    obstaclesList = []

    def __init__(self, obstacles_data_list, vertices_format='axes'):
        for obstacle_id, obstacle in enumerate(obstacles_data_list):
            vertices = []
            if vertices_format == 'coordinates':
                for idx in range(0, len(obstacle[0])):
                    vertices.append((obstacle[0][idx], obstacle[1][idx]))
                height = 50
            elif vertices_format == 'axes':
                vertices = [(obstacle[0], obstacle[1]), (obstacle[0], obstacle[3]),
                            (obstacle[2], obstacle[3]), (obstacle[2], obstacle[1])]
                height = obstacle[-1]

            walls_normal = []
            if vertices_format == 'coordinates':
                n_vertices = len(obstacle[0])
                for wall_idx in range(n_vertices - 1):
                    _wall_vector = np.array(vertices[(wall_idx + 1) % n_vertices]) - np.array(vertices[wall_idx])
                    # test_edge = np.array(vertices[(wall_idx + 2) % n_vertices]) - np.array(vertices[wall_idx])
                    _wall_normal = np.array([_wall_vector[1], -1 * _wall_vector[0]])
                    # if np.dot(_wall_normal, test_edge) > 0:
                    #     _wall_normal = -_wall_normal
                    walls_normal.append(-_wall_normal / np.linalg.norm(_wall_normal))
            self.obstaclesList.append(Obstacle(obstacle_id, height, vertices, walls_normal))

    def check_overlap(self, other_coords):
        for _obstacle in self.obstaclesList:
            if _obstacle.is_overlapping(other_coords.x, other_coords.y, other_coords.z):
                return True
        return False

    def get_total_vertices(self):
        total_vertices = []
        for obstacle in self.obstaclesList:
            total_vertices = total_vertices + obstacle.vertices
        return total_vertices

    def get_total_edges(self):
        edges = []
        for obstacle in self.obstaclesList:
            obstacle_poly = obstacle.vertices + [obstacle.vertices[0]]
            for idx in range(len(obstacle.vertices)):
                edges.append([obstacle_poly[idx], obstacle_poly[idx + 1]])
        return edges

    def print_obstacles(self):
        for obstacle in self.obstaclesList:
            print(obstacle.id, ": ", obstacle.vertices, obstacle.height)

    def plot_obstacles(self, show_flag=False, fill_color=None, polygons=True):
        if not fill_color:
            for obstacle in self.obstaclesList:
                xs, ys = zip(*obstacle.vertices + [obstacle.vertices[0]])
                plt.plot(xs, ys, c='dimgray')
        else:
            if polygons:
                polys = []
                for obstacle in self.obstaclesList:
                    vert_array = np.array(obstacle.vertices)
                    polys.append(Polygon(vert_array, color=fill_color, linewidth=0.00000001))
                return polys
            else:
                rects = []
                for obstacle in self.obstaclesList:
                    xs, ys = zip(*obstacle.vertices + [obstacle.vertices[0]])
                    xs = set(xs)
                    ys = set(ys)
                    corner = min(xs), min(ys)
                    height = max(ys) - min(ys)
                    width = max(xs) - min(xs)
                    rects.append(Rectangle(corner, width, height, color=fill_color))
                return rects

        if show_flag:
            print("SHOWING")
            plt.show()

    def get_boundaries(self):
        x_min, x_max = self.obstaclesList[0].vertices[0][0], self.obstaclesList[0].vertices[0][0]
        y_min, y_max = self.obstaclesList[0].vertices[0][1], self.obstaclesList[0].vertices[0][1]
        for obstacle in self.obstaclesList:
            xs, ys = zip(*obstacle.vertices)
            x_min, x_max = min(x_min, min(xs)), max(x_max, max(xs))
            y_min, y_max = min(y_min, min(ys)), max(y_max, max(ys))
        return [[x_min, x_max], [y_min, y_max]]

    def get_margin_boundary(self, as_polygon=True):
        xs, ys = self.get_boundaries()
        x_min, x_max = xs[0] - BOUNDARY_MARGIN, xs[1] + BOUNDARY_MARGIN
        y_min, y_max = ys[0] - BOUNDARY_MARGIN, ys[1] + BOUNDARY_MARGIN
        if not as_polygon:
            return [x_min, x_max], \
                   [y_min, y_max]
        else:
            return (x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)

    def get_total_segments(self):
        segments = []
        bds = self.get_margin_boundary()
        for i in range(len(bds)):
            segments += [(bds[i], bds[(i + 1) % len(bds)])]
        walls_normals = []
        for obstacle in self.obstaclesList:
            for i in range(len(obstacle.vertices) - 1): #TODO: I removed last vertex here because it's included
                segments += [(obstacle.vertices[i], obstacle.vertices[(i + 1) % len(obstacle.vertices)])]
                walls_normals.append(obstacle.walls_normal[i])
        return segments, walls_normals

    def crop(self, xs, ys):
        idxs_to_pop = []
        for idx, _obs in enumerate(self.obstaclesList):
            for _vert in _obs.vertices:
                if _vert[0] > xs[1] or _vert[0] < xs[0] or _vert[1] > ys[1] or _vert[1] < ys[0]:
                    idxs_to_pop.append(idx)
                    break
        for idx in reversed(idxs_to_pop):
            self.obstaclesList.pop(idx)


class ObstaclesPlotter:
    def plot_obstacles(self, ax):
        bds = self.obstacles_objects.get_margin_boundary(False)
        ax.set_xlim(bds[0][0], bds[0][1])
        ax.set_ylim(bds[1][0], bds[1][1])
        _rects = self.obstacles_objects.plot_obstacles(False, fill_color='gray', polygons=True)
        for _rect in _rects:
            ax.add_patch(_rect)

    @staticmethod
    def show_legend(ax):
        handles, labels = ax.get_legend_handles_labels()

        def flip(items, ncol):
            return iter_chain(*[items[i::ncol] for i in range(ncol)])

        ax.legend(flip(handles, 2), flip(labels, 2), loc='upper left', ncol=2)


# def get_madrid_buildings():
#     new_obstacles = obstacles_madrid_list.copy()
#     if EXTEND_TIMES_FOUR:
#         x_shift_step = 400  # Total dimensions for Madrid Grid is 387 m (east-west) and 552 m (south north).  The
#         # building height is uniformly distributed between 8 and 15 floors with 3.5 m per floor
#         y_shift_step = 570
#         for x_steps in range(0, 3):
#             for y_steps in range(0, 3):
#                 extension = []
#                 for _bldg in obstacles_madrid_list:
#                     extension.append([_bldg[0] + x_steps * x_shift_step, _bldg[1] + y_steps * y_shift_step,
#                                       _bldg[2] + x_steps * x_shift_step, _bldg[3] + y_steps * y_shift_step, _bldg[4]])
#                 # for _bldg in obstacles_madrid_list:
#                 #     extension.append([_bldg[0], _bldg[1] + y_steps * y_shift_step,
#                 #                       _bldg[2], _bldg[3] + y_steps * y_shift_step, _bldg[4]])
#                 # for _bldg in obstacles_madrid_list:
#                 #     extension.append([_bldg[0] + x_steps * x_shift_step,
#                 #                       _bldg[1] + y_steps * y_shift_step, _bldg[2] + x_steps * x_shift_step,
#                 #                       _bldg[3] + y_steps * y_shift_step, _bldg[4]])
#                 new_obstacles += extension
#
#     return Obstacles(new_obstacles)


def get_bologna_buildings():
    _obs_x = np.load(module_dir+'\\xs.npy', allow_pickle=True)
    _obs_y = np.load(module_dir+'\\ys.npy', allow_pickle=True)
    obs_obj = Obstacles(zip(_obs_x, _obs_y), 'coordinates')
    # obs_obj.crop([100, 1200],[400, 1500])
    return obs_obj


if __name__ == '__main__':
    obs_obj = get_bologna_buildings()
    # obs_obj = get_madrid_buildings()
    # obs_obj.crop([100, 1200],[400, 1500])
    _rects = obs_obj.plot_obstacles(True, 'gray', True)
    # _rects = obs_obj.plot_obstacles(False, fill_color='gray')
    bds = obs_obj.get_margin_boundary(False)
    #
    # street_width = 50
    # # street plot
    # rect = Rectangle((bds[0][0] - street_width, bds[1][0]), street_width, bds[1][1] - bds[1][0], color='silver')
    ax = plt.gca()
    # ax.plot(np.linspace(0,2000, 2000, endpoint=False), 1200*np.ones(2000), 'black', '-')
    # ax.plot(np.linspace(0, 2000, 2000, endpoint=False), 1050 * np.ones(2000), 'black', '-')
    # ax.plot(580 * np.ones(2000), np.linspace(0, 2000, 2000, endpoint=False), 'black', '-')
    # ax.plot(710 * np.ones(2000), np.linspace(0, 2000, 2000, endpoint=False), 'black', '-')
    # ax.add_patch(rect)
    #
    for _rect in _rects:
        ax.add_patch(_rect)
    # # xs = np.array([0, 15.0, 200.0])
    # # ys = np.array([0.0, 390.0, 370.0])
    # # plt.plot(xs, ys, c='red')
    # # plt.plot([], [], c='red', label='FSO link')
    # # plt.plot(250.0, 380.0, c='green', marker='o', label='Hotspot center', linestyle='none')
    # # # plt.plot(200.0, 370.0, c='green', marker='o', label='Hotspot center', linestyle='none')
    #
    # # xs = [82.5, 165, 82.5, 165, 345.5, 263, 345.5, 263]
    # # ys = [113.5, 227, 479.5, 366, 113.5, 227, 479.5, 366]
    # #
    # # for x, y in zip(xs, ys):
    # #     plt.plot(x, y, c='green', marker='o', linestyle='none')
    # #     # circle = plt.Circle((x, y), 300, color='r')
    # #     # ax = plt.gca()
    # #     # ax.add_patch(circle)
    #
    # # plt.legend(loc="upper left")
    plt.ylim([bds[1][0], bds[1][1]])
    plt.xlim([bds[0][0], bds[0][1]])

    plt.show()
    # plt.savefig('bologna.eps', format='eps')

# _obs = obs_obj.obstaclesList[10]
# n_vertices = len(_obs.vertices)
# _obs.walls_normal = []
# for wall_idx in range( n_vertices -1):
#     _wall_vector = np.array(_obs.vertices[(wall_idx+1)%n_vertices]) - np.array(_obs.vertices[wall_idx])
#     test_edge =  np.array(_obs.vertices[(wall_idx+2)%n_vertices]) -  np.array(_obs.vertices[wall_idx])
#     _wall_normal = np.array([_wall_vector[1], -1 * _wall_vector[0]])
#     # if np.dot(_wall_normal, test_edge) > 0:
#     #     _wall_normal = -_wall_normal
#     _obs.walls_normal.append(-_wall_normal/np.linalg.norm(_wall_normal))

# xs, ys = zip(*_obs.vertices)
# plt.plot(xs, ys, c='dimgray')
# for idx, _wall_normal in enumerate(_obs.walls_normal):
#     xs, ys = zip(*[_obs.vertices[idx], _obs.vertices[idx+1]])
#     plt.plot(xs+_wall_normal[0], ys+_wall_normal[1], c='black')
# plt.show()