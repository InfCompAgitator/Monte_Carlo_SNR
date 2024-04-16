from src.math_tools import get_azimuth, get_elevation
from src.data_structures import Coords3d
import numpy as np
from src.parameters import CARRIER_FREQ, SPEED_OF_LIGHT, TRANSMISSION_POWER, RECEIVER_SENSITIVITY, SIGMA_SHADOWING, \
    ANTENNA_GAIN, K_0, rng, SIGMA_SHADOWING, MAXIMUM_BEAMFORMING_THETA
from scipy.special import erfc
from math import factorial
from src.math_tools import db2lin
from scipy.stats import poisson
from math import copysign
from scipy.special import comb, binom


def calculate_beam_access_probability(v_density, fp, dbs_loc: Coords3d, beam_length, beam_width):
    dist_to_center = dbs_loc.get_distance_to(fp.center_coords)
    dist_to_start = dbs_loc.get_distance_to(fp.start_coords)
    dist_to_end = dbs_loc.get_distance_to(fp.end_coords)
    farthest_dist = max(dist_to_start, dist_to_end)
    if np.degrees(np.arctan(dbs_loc.z/farthest_dist)) > MAXIMUM_BEAMFORMING_THETA:
        return 1, 0, 1, 1, 1, 1, 1, 1
    cos_theta = (dist_to_start ** 2 + dist_to_end ** 2 - beam_length ** 2) / (2 * dist_to_start * dist_to_end)
    phi = abs(2 * np.arctan(beam_width / dist_to_center))
    theta = np.arccos(cos_theta)
    beam_forming_gain = 41000 / (phi * theta)
    # beam_forming_gain = 100000000

    N_STEPS = 200
    del_x_y = fp.start_coords - fp.end_coords
    if del_x_y.x != 0:
        direction = del_x_y.y / del_x_y.x
        d_x_1 = np.sqrt(((beam_length / N_STEPS) ** 2) / (1 + direction ** 2)) * copysign(1, del_x_y.x)
        d_y_1 = direction * d_x_1
    else:
        d_x_1 = 0
        d_y_1 = beam_length / N_STEPS

    outage_probs = np.zeros(N_STEPS + 1)
    sampling_coords = [fp.start_coords + i * Coords3d(d_x_1, d_y_1, 0) for i in range(N_STEPS + 1)]
    for idx, _coord in enumerate(sampling_coords):
        dist = dbs_loc.get_distance_to(_coord)
        fspl = 20 * np.log10(4 * np.pi * CARRIER_FREQ * dist / SPEED_OF_LIGHT)
        rx_power = ANTENNA_GAIN + TRANSMISSION_POWER + 10 * np.log10(beam_forming_gain) - fspl
        outage_probs[idx] = 1 - 0.5 * erfc((RECEIVER_SENSITIVITY - rx_power) / (np.sqrt(2) * SIGMA_SHADOWING))
    step_out_prob_mean = outage_probs.mean()

    fspl = 20 * np.log10(4 * np.pi * CARRIER_FREQ * dist_to_center / SPEED_OF_LIGHT)
    fspl_end = 20 * np.log10(4 * np.pi * CARRIER_FREQ * dist_to_end / SPEED_OF_LIGHT)
    fspl_start = 20 * np.log10(4 * np.pi * CARRIER_FREQ * dist_to_start / SPEED_OF_LIGHT)

    received_power_end = ANTENNA_GAIN + TRANSMISSION_POWER + 10 * np.log10(beam_forming_gain) - fspl_end
    outage_probability_end = 1 - 0.5 * erfc(
        (RECEIVER_SENSITIVITY - received_power_end) / (np.sqrt(2) * SIGMA_SHADOWING))
    received_power_start = ANTENNA_GAIN + TRANSMISSION_POWER + 10 * np.log10(beam_forming_gain) - fspl_start
    outage_probability_start = 1 - 0.5 * erfc(
        (RECEIVER_SENSITIVITY - received_power_start) / (np.sqrt(2) * SIGMA_SHADOWING))
    received_power = ANTENNA_GAIN + TRANSMISSION_POWER + 10 * np.log10(beam_forming_gain) - fspl
    outage_probability = 1 - 0.5 * erfc((RECEIVER_SENSITIVITY - received_power) / (np.sqrt(2) * SIGMA_SHADOWING))

    beam_v_density = beam_length * v_density
    prob_at_least_one_event = 1 - poisson.pmf(0, beam_v_density)

    # My way
    p_sum = 0
    prev_sum = p_sum
    for k in range(int(np.floor(K_0) + 1), 1000):
        poisson_n = 1 / factorial(k) * beam_v_density ** k * np.exp(-beam_v_density)
        outage_prob_sum = 0
        for k_non_outaged in range(int(np.floor(K_0) + 1), k + 1):
            k_outaged = k - k_non_outaged
            outage_prob_sum += (1 - step_out_prob_mean) ** k_non_outaged * step_out_prob_mean ** k_outaged * comb(k - 1,
                                                                                                                  k_non_outaged - 1)
        p_sum += poisson_n * outage_prob_sum
        if abs(prev_sum - p_sum) < 1e-100:
            break
        prev_sum = p_sum
    not_access_prob_new_1 = prev_sum
    not_access_prob_new_2 = prev_sum / prob_at_least_one_event

    # Francesca's
    # beam_v_density = beam_v_density * (1 - step_out_prob_mean)
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

    not_access_prob_old = 1 - poisson.cdf(np.floor(K_0), beam_v_density * (1 - step_out_prob_mean))

    return not_access_prob_old, beam_forming_gain, outage_probability, outage_probability_end, \
        outage_probability_start, step_out_prob_mean, not_access_prob_new_1, not_access_prob_new_2


def get_received_power_sample(v_coords, dbs_coord, beam_forming_gain):
    dist = v_coords.get_distance_to(dbs_coord)
    fspl = 20 * np.log10(4 * np.pi * CARRIER_FREQ * dist / SPEED_OF_LIGHT)
    received_power = ANTENNA_GAIN + TRANSMISSION_POWER + 10 * np.log10(beam_forming_gain) - fspl - rng.normal(0,
                                                                                                              SIGMA_SHADOWING)
    return received_power
