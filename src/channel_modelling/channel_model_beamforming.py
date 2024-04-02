from src.math_tools import get_azimuth, get_elevation
from src.data_structures import Coords3d
import numpy as np
from src.parameters import CARRIER_FREQ, SPEED_OF_LIGHT, TRANSMISSION_POWER, RECEIVER_SENSITIVITY, SIGMA_SHADOWING, \
    ANTENNA_GAIN, K_0, rng, SIGMA_SHADOWING
from scipy.special import erfc
from math import factorial
from src.math_tools import db2lin
from scipy.stats import poisson


def calculate_beam_access_probability(v_density, fp, dbs_loc: Coords3d, beam_length, beam_width):
    dist_to_center = dbs_loc.get_distance_to(fp.center_coords)
    dist_to_start = dbs_loc.get_distance_to(fp.start_coords)
    dist_to_end = dbs_loc.get_distance_to(fp.end_coords)
    cos_theta = (dist_to_start ** 2 + dist_to_end ** 2 - beam_length ** 2) / (2 * dist_to_start * dist_to_end)
    phi = abs(2 * np.arctan(beam_width / dist_to_center))
    theta = np.arccos(cos_theta)
    beam_forming_gain = 41000 / (phi * theta)
    beam_forming_gain = 1000
    # if beam_forming_gain > 10000:
    #     beam_forming_gain = 10000
    fspl = 20 * np.log10(4 * np.pi * CARRIER_FREQ * dist_to_center / SPEED_OF_LIGHT)
    fspl_end = 20 * np.log10(4 * np.pi * CARRIER_FREQ * dist_to_end / SPEED_OF_LIGHT)
    fspl_start = 20 * np.log10(4 * np.pi * CARRIER_FREQ * dist_to_start / SPEED_OF_LIGHT)
    received_power_end = ANTENNA_GAIN + TRANSMISSION_POWER + 10 * np.log10(beam_forming_gain) - fspl_end
    outage_probability_end = 1 - 0.5 * erfc((RECEIVER_SENSITIVITY - received_power_end) / (np.sqrt(2) * SIGMA_SHADOWING))
    received_power_start = ANTENNA_GAIN + TRANSMISSION_POWER + 10 * np.log10(beam_forming_gain) - fspl_start
    outage_probability_start = 1 - 0.5 * erfc(
        (RECEIVER_SENSITIVITY - received_power_start) / (np.sqrt(2) * SIGMA_SHADOWING))
    received_power = ANTENNA_GAIN + TRANSMISSION_POWER + 10 * np.log10(beam_forming_gain) - fspl
    outage_probability = 1 - 0.5 * erfc((RECEIVER_SENSITIVITY - received_power) / (np.sqrt(2) * SIGMA_SHADOWING))
    # if outage_probability > 0.99:
    #     return 1, 1, beam_forming_gain, outage_probability
    beam_v_density = beam_length * v_density
    # My way
    # p_sum = 0
    # prev_sum = p_sum
    # for k in range(int(np.floor(K_0 + 1)), 1000):
    #     try:
    #         p_sum += beam_v_density ** (np.floor(k / (1 - outage_probability))) * np.exp(-beam_v_density) / factorial(
    #             int(np.ceil(k / (1 - outage_probability))))
    #     except:
    #         break
    #     if abs(prev_sum - p_sum) < 1e-10:
    #         break
    #     prev_sum = p_sum
    # my_way = prev_sum
    my_way = 1 - poisson.cdf(K_0 / (1 - outage_probability), 1/beam_v_density)
    # Francesca's
    beam_v_density = beam_v_density * (1 - outage_probability)
    # p_sum = 0
    # prev_sum = p_sum
    # for k in range(int(np.floor(K_0 + 1)), 1000):
    #     try:
    #         p_sum += beam_v_density ** k * np.exp(-beam_v_density) / factorial(int(k))
    #     except:
    #         break
    #     if abs(prev_sum - p_sum) < 1e-10:
    #         break
    #     prev_sum = p_sum
    prev_sum = 1 - poisson.cdf(K_0, 1/beam_v_density)
    return my_way, prev_sum, beam_forming_gain, outage_probability, outage_probability_end, outage_probability_start


def get_received_power_sample(v_coords, dbs_coord, beam_forming_gain):
    dist = v_coords.get_distance_to(dbs_coord)
    fspl = 20 * np.log10(4 * np.pi * CARRIER_FREQ * dist / SPEED_OF_LIGHT)
    received_power = ANTENNA_GAIN + TRANSMISSION_POWER + 10 * np.log10(beam_forming_gain) - fspl - rng.normal(0,
                                                                                                              SIGMA_SHADOWING)
    return received_power
