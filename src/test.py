import numpy as np

from src.bologna.streets import Street, Vehicle
from src.data_structures import Coords3d
from src.parameters import DBS_COVERAGE_LENGTH, VEH_SPEED_RANGE, rng, BEAM_LENGTH, STREET_WIDTH, K_0, \
    RECEIVER_SENSITIVITY, DBS_HEIGHT
import heapq
from src.bologna.city import Footprint
from src.channel_modelling.channel_model_beamforming import calculate_beam_access_probability, get_received_power_sample
from tqdm import tqdm
from scipy.stats import poisson

import matplotlib
# matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt
T_STEP = 1
T_END = 1000000
DENSITY = 0.4


def generate_vehicle_arrival_times(lambda_val, T_end=T_END, distance=BEAM_LENGTH):
    arrival_times = []
    t = 0
    t_shift = distance / VEH_SPEED_RANGE[0]
    arrival_t_end = T_end + t_shift
    while t < arrival_t_end:
        interarrival_time = rng.exponential(
            1 / lambda_val)  # Generate interarrival time from exponential distribution
        t += interarrival_time  # Update time and append arrival time to list
        arrival_times.append(t)
        # print("arrival times: ", arrival_times)
    return arrival_times


def run_sim_1(height, density):
    street_1 = Street(0, 1, Coords3d(0, 0, 0), Coords3d(0, BEAM_LENGTH, 0))
    dbs_loc = (street_1.start_node_coords + street_1.end_node_coords) / 2
    dbs_loc.z = height
    fp = Footprint(Coords3d(0, BEAM_LENGTH, 0) / 2, start_coords=street_1.start_node_coords,
                   end_coords=street_1.end_node_coords)
    v_density_meter = density / VEH_SPEED_RANGE[0]
    (not_acces_probability, b_f_gain, math_outage_probability, math_outage_probability_end,
     math_outage_probability_start,
     math_out_probs_steps, not_access_probability_new_1,
     not_access_probability_new_2) = calculate_beam_access_probability(v_density_meter, fp, dbs_loc, BEAM_LENGTH,
                                                                       STREET_WIDTH)
    _veh_arrival_times = generate_vehicle_arrival_times(lambda_val=density)
    vehicles = []
    n_vs = 0
    one_arrived = False
    avg_i, avg = 0, 0
    meas_access_probability = 0
    meas_outage_probability = 0.0
    meas_outage_probability_i = 0
    for t in range(0, T_END):
        n_vs_satisfied_snr = 0
        while _veh_arrival_times[0] < t:
            new_t = heapq.heappop(_veh_arrival_times)

            new_veh = Vehicle(street_1.start_node_coords, 1, 0)
            new_veh.set_end_point(street_1.end_node_coords.copy())
            new_veh.update_coords(t - new_t)
            vehicles.append(new_veh)
            n_vs += 1
        for idx in reversed(range(len(vehicles))):
            _v = vehicles[idx]
            rx_power = get_received_power_sample(_v.coords, dbs_loc, b_f_gain)
            if rx_power >= RECEIVER_SENSITIVITY:
                n_vs_satisfied_snr += 1
            if _v.update_coords(1):
                one_arrived = True
                vehicles.pop(idx)
                n_vs -= 1
                del _v

        if one_arrived and _veh_arrival_times != []:
            avg_i += 1
            avg = avg + (n_vs / street_1.length - avg) / avg_i
            vehicles_density = avg
            if n_vs > 0:
                meas_outage_probability_i += 1
                pout = 1 - n_vs_satisfied_snr / n_vs
                meas_outage_probability = meas_outage_probability + (
                        pout - meas_outage_probability) / meas_outage_probability_i
                if n_vs_satisfied_snr > K_0 or n_vs_satisfied_snr == 0:
                    meas_access_probability = (meas_access_probability -
                                               meas_access_probability / meas_outage_probability_i)
                else:
                    meas_access_probability = (meas_access_probability +
                                               ((1 - pout) - meas_access_probability) / meas_outage_probability_i)

    return (meas_access_probability, (1 - not_acces_probability) * (
            1 - math_out_probs_steps), 1 - not_acces_probability, math_outage_probability,
            meas_outage_probability, math_outage_probability_end, math_outage_probability_start, math_out_probs_steps,
            vehicles_density, 1 - not_access_probability_new_1 - math_out_probs_steps,
            1 - not_access_probability_new_2 - math_out_probs_steps)


if __name__ == '__main__':

    # street_1 = Street(0, 1, Coords3d(0, 0, 0), Coords3d(0, BEAM_LENGTH, 0))
    # dbs_loc = (street_1.start_node_coords + street_1.end_node_coords) / 2
    # dbs_loc.z = 100
    # fp = Footprint(Coords3d(0, BEAM_LENGTH, 0) / 2, start_coords=street_1.start_node_coords,
    #                end_coords=street_1.end_node_coords)
    # v_density_meter = DENSITY / VEH_SPEED_RANGE[0]
    #
    # poss_heights = np.arange(10, 3000, 10)
    # b_f_gain, outage_probability = np.zeros_like(poss_heights, dtype=np.float32), np.zeros_like(poss_heights, dtype=np.float32)
    # for idx, _height in enumerate(poss_heights):
    #     dbs_loc.y = _height
    #     _, _, b_f_gain[idx], outage_probability[idx] = calculate_beam_access_probability(v_density_meter, fp, dbs_loc,
    #                                                                               BEAM_LENGTH, STREET_WIDTH)
    #
    # fig, ax = plt.subplots()
    # ax.plot(poss_heights, outage_probability, label='outage probability')
    # fig.legend(loc='upper left')
    # # ax.plot(poss_heights, b_f_gain, label='beamforming gain')
    # fig.show()
    density = 0.4
    poss_heights = np.arange(10, 100, 10)
    meas_access_probability = np.zeros_like(poss_heights, dtype=np.float32)
    math_access_probability = np.zeros_like(poss_heights, dtype=np.float32)
    math_access_probability_new_1 = np.zeros_like(poss_heights, dtype=np.float32)
    math_access_probability_new_2 = np.zeros_like(poss_heights, dtype=np.float32)
    math_outage_probability = np.zeros_like(poss_heights, dtype=np.float32)
    math_outage_probability_end = np.zeros_like(poss_heights, dtype=np.float32)
    math_outage_probability_steps = np.zeros_like(poss_heights, dtype=np.float32)
    meas_outage_probability = np.zeros_like(poss_heights, dtype=np.float32)
    math_not_allocation_probability = np.zeros_like(poss_heights, dtype=np.float32)
    vehicles_density = np.zeros_like(poss_heights, dtype=np.float32)
    idx = 0
    for _height in tqdm(poss_heights):
        meas_access_probability[idx], math_access_probability[idx], math_not_allocation_probability[idx], \
            math_outage_probability[idx], meas_outage_probability[idx], math_outage_probability_end[idx], _, \
            math_outage_probability_steps[idx], vehicles_density[idx], math_access_probability_new_1[idx], \
            math_access_probability_new_2[idx] = run_sim_1(_height, density)
        idx += 1
    fig, ax = plt.subplots()
    ax.plot(poss_heights, meas_access_probability, label="Measured Access P.", linestyle="-.")
    ax.plot(poss_heights, math_access_probability, label="Math Access P. Old", linestyle="-", alpha=0.5)
    ax.plot(poss_heights, math_access_probability_new_1, label="Math Access P. New 1", linestyle=":", alpha=0.5)
    ax.plot(poss_heights, math_access_probability_new_2, label="Math Access P. New 2", linestyle=":", alpha=0.5)
    ax.set_title(f'Density = {density}', loc='left')
    fig.legend(loc=(0.32, 0.88), ncol=2)
    ax.set_xlabel("Drone Height [m]")
    ax.set_ylabel("Access Probability")
    fig.show()
    fig.legend()

    fig_pout, ax_pout = plt.subplots()
    ax_pout.plot(poss_heights, math_outage_probability, label="Math Pout at Center", linestyle="--")
    ax_pout.plot(poss_heights, math_outage_probability_end, label="Math Pout at Endpoint", linestyle=":")
    ax_pout.plot(poss_heights, (math_outage_probability + math_outage_probability_end) / 2,
                 label="Average Pout of Center and Endpoint", linestyle="--")
    ax_pout.plot(poss_heights, meas_outage_probability, label="Measured Pout", linestyle="-")
    ax_pout.plot(poss_heights, math_outage_probability_steps, label="Average Pout Across Street Steps",
                 linestyle=":")
    ax_pout.set_xlabel("Drone Height [m]")
    ax_pout.set_ylabel("Outage Probability")
    ax_pout.set_title(f'Density = {density}')
    fig_pout.legend()
    fig_pout.show()

    # street_1 = Street(0, 1, Coords3d(0, 0, 0), Coords3d(0, BEAM_LENGTH, 0))
    # dbs_loc = (street_1.start_node_coords + street_1.end_node_coords) / 2
    # dbs_loc.z = DBS_HEIGHT
    # # fps_start = [Coords3d(0, i * BEAM_LENGTH, 0) for i in range(int(DBS_COVERAGE_LENGTH / BEAM_LENGTH))]
    # # fps_end = [Coords3d(0, i * BEAM_LENGTH, 0) for i in range(1, int(DBS_COVERAGE_LENGTH / BEAM_LENGTH) + 1)]
    # # fps = [Footprint(center_coords=(start + end) / 2, start_coords=start, end_coords=end) for start, end in zip(fps_start, fps_end)]
    # fp = Footprint(Coords3d(0, BEAM_LENGTH, 0)/2, start_coords=street_1.start_node_coords, end_coords=street_1.end_node_coords)
    # v_density_meter = DENSITY / VEH_SPEED_RANGE[0]
    # my_way, a_p, b_f_gain, outage_probability = calculate_beam_access_probability(v_density_meter, fp, dbs_loc, BEAM_LENGTH, STREET_WIDTH)
    # _veh_arrival_times = generate_vehicle_arrival_times()
    # vehicles = []
    # n_vs = 0
    # n_vs_satisfied_snr = 0
    # one_arrived = False
    # avg_i, avg = 0, 0
    # avg_satisfied = 0
    # for t in range(0, T_END):
    #     n_vs_satisfied_snr = 0
    #     while _veh_arrival_times[0] < t:
    #         new_t = heapq.heappop(_veh_arrival_times)
    #         new_veh = Vehicle(street_1.start_node_coords, 1, 0)
    #         new_veh.set_end_point(street_1.end_node_coords)
    #         new_veh.update_coords(t - new_t)
    #         vehicles.append(new_veh)
    #         n_vs += 1
    #     for idx, _v in enumerate(vehicles):
    #         rx_power = get_received_power_sample(_v.coords, dbs_loc, b_f_gain)
    #         if rx_power >= RECEIVER_SENSITIVITY:
    #             n_vs_satisfied_snr += 1
    #         if _v.update_coords(1):
    #             one_arrived = True
    #             vehicles.pop(idx)
    #             n_vs -= 1
    #             del _v
    #
    #     if one_arrived and _veh_arrival_times != []:
    #         avg_i += 1
    #         avg = avg + (n_vs / street_1.length - avg) / avg_i
    #         # print(avg)
    #         if n_vs_satisfied_snr > K_0:
    #             avg_satisfied = avg_satisfied - avg_satisfied/ avg_i
    #         else:
    #             avg_satisfied = avg_satisfied + (1 - avg_satisfied) / avg_i
    #         print(avg_satisfied)
    #
    # print('-----------------------------------')
    # print( 1 - a_p *  (1- outage_probability))
    # print(1 - a_p)
    # print(1 - (outage_probability * a_p))
    # print('-----------------------------------')
    #
    # print(1 - my_way * (1 - outage_probability))
    # print(1 - my_way)
    # print(1 - (outage_probability * my_way))
