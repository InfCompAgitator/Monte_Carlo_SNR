import visibility_polygon
from src.data_structures import Coords3d
from itertools import repeat, compress
from multiprocessing import Pool


def get_vis_polygon(obstacles_segments: list, coords_x, coords_y) -> list:
    return visibility_polygon.get_visibility(obstacles_segments, [coords_x, coords_y])


def check_if_point_in_polygon(input_poly: list, test_pt: tuple) -> bool:
    if visibility_polygon.point_in_polygon(input_poly, test_pt):
        return True
    else:
        return False


def get_vertices_in_polygon(poly_in, vertices_list):
    # with Pool(1) as p:
    #     results = list(
    #         p.starmap(check_if_point_in_polygon, zip(repeat(poly_in), vertices_list), chunksize=100))
    # return list(compress(vertices_list, results))
    results = [check_if_point_in_polygon(poly_in, vert) for vert in vertices_list]
    return vertices_list, results
