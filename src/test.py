import numpy as np

from src.bologna.streets import Street, Vehicle
from src.data_structures import Coords3d
from src.parameters import DBS_COVERAGE_LENGTH, VEH_SPEED_RANGE, rng, BEAM_LENGTH, STREET_WIDTH ,K_0, RECEIVER_SENSITIVITY, DBS_HEIGHT
import heapq
from src.bologna.city import Footprint
from src.channel_modelling.channel_model_beamforming import calculate_beam_access_probability, get_received_power_sample
from tqdm import tqdm

import matplotlib
# matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt



T_END = 1000
DENSITY = 0.1
def generate_vehicle_arrival_times(lambda_val=DENSITY, T_end=T_END, distance=BEAM_LENGTH):
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

def run_sim_1(height):
    street_1 = Street(0, 1, Coords3d(0, 0, 0), Coords3d(0, BEAM_LENGTH, 0))
    dbs_loc = (street_1.start_node_coords + street_1.end_node_coords) / 2
    dbs_loc.z = height
    # fps_start = [Coords3d(0, i * BEAM_LENGTH, 0) for i in range(int(DBS_COVERAGE_LENGTH / BEAM_LENGTH))]
    # fps_end = [Coords3d(0, i * BEAM_LENGTH, 0) for i in range(1, int(DBS_COVERAGE_LENGTH / BEAM_LENGTH) + 1)]
    # fps = [Footprint(center_coords=(start + end) / 2, start_coords=start, end_coords=end) for start, end in zip(fps_start, fps_end)]
    fp = Footprint(Coords3d(0, BEAM_LENGTH, 0) / 2, start_coords=street_1.start_node_coords,
                   end_coords=street_1.end_node_coords)
    v_density_meter = DENSITY / VEH_SPEED_RANGE[0]
    my_way, a_p, b_f_gain, outage_probability, outage_probability_end, outage_probability_start = calculate_beam_access_probability(v_density_meter, fp, dbs_loc,
                                                                                  BEAM_LENGTH, STREET_WIDTH)
    _veh_arrival_times = generate_vehicle_arrival_times()
    vehicles = []
    n_vs = 0
    n_vs_satisfied_snr = 0
    one_arrived = False
    avg_i, avg = 0, 0
    avg_satisfied = 0
    avg_pout = 0.0
    avg_pout_i = 0
    for t in range(0, T_END):
        n_vs_satisfied_snr = 0
        while _veh_arrival_times[0] < t:
            new_t = heapq.heappop(_veh_arrival_times)

            new_veh = Vehicle(street_1.start_node_coords, 1, 0)
            new_veh.set_end_point(street_1.end_node_coords.copy())
            new_veh.update_coords(t - new_t)
            vehicles.append(new_veh)
            n_vs += 1
        for idx, _v in enumerate(vehicles):
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
            # print(avg)
            if n_vs>0:
                avg_pout_i += 1
                pout = 1 - n_vs_satisfied_snr/ n_vs
                avg_pout = avg_pout + (pout - avg_pout)/ avg_pout_i
            if n_vs_satisfied_snr > K_0 or n_vs_satisfied_snr==0:
                avg_satisfied = avg_satisfied - avg_satisfied / avg_i
            else:
                avg_satisfied = avg_satisfied + (1 - avg_satisfied) / avg_i

    return avg_satisfied, (1 - a_p * (1 - outage_probability)),  a_p, (1 - outage_probability) * ( 1 - a_p),(1 - my_way * (1 - outage_probability)),\
    my_way, (1 - outage_probability)*(1 - my_way), outage_probability, avg_pout, outage_probability_end, outage_probability_start


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


    poss_heights = np.arange(10, 100,10)
    res_1_1 = np.zeros_like(poss_heights, dtype=np.float32)
    res_1_2 = np.zeros_like(poss_heights, dtype=np.float32)
    res_1_3 = np.zeros_like(poss_heights, dtype=np.float32)
    res_2_1 = np.zeros_like(poss_heights, dtype=np.float32)
    res_2_2 = np.zeros_like(poss_heights, dtype=np.float32)
    res_2_3 = np.zeros_like(poss_heights, dtype=np.float32)
    sim_res = np.zeros_like(poss_heights, dtype=np.float32)
    outage_probability = np.zeros_like(poss_heights, dtype=np.float32)
    outage_probability_end = np.zeros_like(poss_heights, dtype=np.float32)
    avg_pout = np.zeros_like(poss_heights, dtype=np.float32)
    idx = 0
    for _height in tqdm(poss_heights):
        a = run_sim_1(_height)
        sim_res[idx], res_1_1[idx], res_1_2[idx], res_1_3[idx], res_2_1[idx], res_2_2[idx], res_2_3[idx], outage_probability[idx], avg_pout[idx], outage_probability_end[idx], _ = a
        idx += 1
    fig, ax = plt.subplots()
    # ax.plot(poss_heights, res_1_1, label="Res1_1")
    # ax.plot(poss_heights, res_1_2, label="Not Access 1")
    ax.plot(poss_heights, res_1_3, label="Res1_3")
    # ax.plot(poss_heights, res_2_1, label="Res2_1")
    # ax.plot(poss_heights, res_2_2, label="Not Access 2")
    # ax.plot(poss_heights, res_2_3, label="Res2_3")
    ax.plot(poss_heights, sim_res, label="sim", linestyle="--")
    ax.plot(poss_heights, outage_probability, label="outage_prob", linestyle="-.")
    ax.plot(poss_heights, outage_probability_end, label="outage_prob_end", linestyle=":")
    ax.plot(poss_heights, (outage_probability + outage_probability_end)/2, label="outage_prob_avg", linestyle=":")
    ax.plot(poss_heights, avg_pout, label="meas_outage_prob", linestyle="-.")
    fig.legend()
    fig.show()

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
