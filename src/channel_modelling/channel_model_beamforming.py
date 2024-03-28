from src.math_tools import get_azimuth, get_elevation
from src.data_structures import Coords3d
import numpy as np
def calculate_beam_access_probability(v_density, fp, dbs_loc: Coords3d, beam_length, beam_width):
    dist_to_center = dbs_loc.get_distance_to(fp.center_coords)
    dist_to_start = dbs_loc.get_distance_to(fp.start_coords)
    dist_to_end = dbs_loc.get_distance_to(fp.end_coords)
    cos_theta = (dist_to_start**2 + dist_to_end**2 - beam_length**2)/(2*dist_to_start*dist_to_end)
    phi = abs(2 * np.arctan(beam_width / dist_to_center))
    theta = np.arccos(cos_theta)
    beam_forming_gain = 41000/(phi*theta)


    pass
