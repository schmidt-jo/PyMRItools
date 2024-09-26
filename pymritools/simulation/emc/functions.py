import torch
from scipy import stats
import numpy as np
from emc_torch import options


def pulse_calibration_integral(sim_params: options.SimulationParameters,
                               excitation: bool,
                               refocusing_pulse_number: int = 0) -> torch.tensor:
    """
    Calibrates pulse waveform for given flip angle, adds phase if given
    """
    # get b1 values - error catch again if single value is given
    b1_vals = options.SimulationData.build_array_from_list_of_lists_args(sim_params.settings.b1_list)
    # if not isinstance(b1_vals, list):
    #     b1_vals = [b1_vals]
    # b1_vals = torch.tensor(b1_vals)
    if excitation:
        angle_flip = sim_params.sequence.excitation_angle
        phase = sim_params.sequence.excitation_phase / 180.0 * torch.pi
    else:
        # excitation pulse always 0th pulse
        angle_flip = sim_params.sequence.refocus_angle[refocusing_pulse_number - 1]
        phase = sim_params.sequence.refocus_phase[refocusing_pulse_number - 1] / 180.0 * torch.pi
    # calculate with applied actual flip angle offset
    sim_params.pulse.set_flip_angle(flip_angle_rad=angle_flip / 180 * torch.pi, excitation=excitation)
    b1_pulse = torch.from_numpy(sim_params.pulse.get_pulse(excitation=excitation).amplitude) * torch.exp(
        1j * (torch.from_numpy(sim_params.pulse.get_pulse(excitation=excitation).phase) + phase)
    )
    b1_pulse_calibrated = b1_pulse[None, :] * b1_vals[:, None]
    return b1_pulse_calibrated


def propagate_matrix_mag_vector(propagation_matrix: torch.tensor, sim_data: options.SimulationData):
    """
    we setup the simulation around propagation matrices with dimensions:
        [t1s: i, t2s: j, b1s: k, samples: l, 4: m, 4: n]
    The propagation vector consequently propagates in 4d for all dictionary entries
        [t1s: i, t2s: j, b1s:k, samples: l, 4: n]
    this function just scripts the matrix multiplication
    """
    sim_data.magnetization_propagation = torch.einsum('ijklmn, ijkln -> ijklm',
                                                      propagation_matrix,
                                                      sim_data.magnetization_propagation)
    return sim_data


def propagate_gradient_pulse_relax(
        pulse_x: torch.tensor,
        pulse_y: torch.tensor,
        grad: torch.tensor,
        sim_data: options.SimulationData,
        dt_s: torch.tensor) -> torch.tensor:
    """
    Calculating the effect of grad pulse data on magnetization propagation through iterative matrix multiplication
    :param pulse_x: torch tensor holding x dimension data of pulse - dim [b1, dt]
    :param pulse_y: torch tensor holding y dimension data of pulse - dim [b1, dt]
    :param grad: torch tensor holding slice gradient amplitude data - dim [dt]
    :param sim_data: simulation data object carrying simulation values through sim
    :param dt_s: torch tensor holding sampling step size in seconds
    :return:
    """
    # get pulse matrices per step and propagate
    # iterate through the pulse grad shape and multiply the matrix propagators
    # dims [num_t1s, num_t2s, num_b1s, num_samples, 4, 4]
    # check if dimensions are right, assumed dimensions given in doc string, if pulse data is for 1 b1 value only,
    # we can fix tensor dimensions
    if torch.prod(torch.tensor(pulse_x.shape)) < 2:
        iter_range = 1
        if pulse_x.shape.__len__() < 2:
            pulse_x = pulse_x[None, :]
            pulse_y = pulse_y[None, :]
    else:
        iter_range = pulse_x.shape[1]
    # iterate through time points
    for i in range(iter_range):
        prop_step = matrix_propagation_grad_pulse_multidim(
            pulse_x=pulse_x[:, i], pulse_y=pulse_y[:, i], grad=grad[i], dt_s=dt_s, sim_data=sim_data
        )
        # propagation_matrix = torch.matmul(
        #     prop_step,
        #     propagation_matrix
        # )
        sim_data = propagate_matrix_mag_vector(prop_step, sim_data=sim_data)
    return sim_data


def setup_rot_mats(angle: torch.tensor):
    if angle.shape.__len__() < 1:
        a_shape = 1
    else:
        a_shape = angle.shape
    device = angle.device
    t_0 = torch.zeros(a_shape).to(device)
    t_1 = torch.ones(a_shape).to(device)
    return a_shape, t_0, t_1


def rotation_matrix_x(angle: torch.tensor):
    a_shape, t_0, t_1 = setup_rot_mats(angle)
    rot_matrix = torch.stack([
        torch.stack([t_1, t_0, t_0, t_0]),
        torch.stack([t_0, torch.cos(angle), -torch.sin(angle), t_0]),
        torch.stack([t_0, torch.sin(angle), torch.cos(angle), t_0]),
        torch.stack([t_0, t_0, t_0, t_1])
    ])
    rot_matrix = torch.movedim(rot_matrix, -1, 0)
    if a_shape.__len__() > 1:
        rot_matrix = torch.movedim(rot_matrix, -1, 0)
    return rot_matrix


def rotation_matrix_y(angle: torch.tensor):
    a_shape, t_0, t_1 = setup_rot_mats(angle)
    rot_matrix = torch.stack([
        torch.stack([torch.cos(angle), t_0, torch.sin(angle), t_0]),
        torch.stack([t_0, t_1, t_0, t_0]),
        torch.stack([-torch.sin(angle), t_0, torch.cos(angle), t_0]),
        torch.stack([t_0, t_0, t_0, t_1])
    ])
    rot_matrix = torch.movedim(rot_matrix, -1, 0)
    if a_shape.__len__() > 1:
        rot_matrix = torch.movedim(rot_matrix, -1, 0)
    return rot_matrix


def rotation_matrix_z(angle: torch.tensor):
    a_shape, t_0, t_1 = setup_rot_mats(angle)
    rot_matrix = torch.stack([
        torch.stack([torch.cos(angle), -torch.sin(angle), t_0, t_0]),
        torch.stack([torch.sin(angle), torch.cos(angle), t_0, t_0]),
        torch.stack([t_0, t_0, t_1, t_0]),
        torch.stack([t_0, t_0, t_0, t_1])
    ])
    rot_matrix = torch.movedim(rot_matrix, -1, 0)
    if a_shape.__len__() > 1:
        rot_matrix = torch.movedim(rot_matrix, -1, 0)
    return rot_matrix


def matrix_propagation_grad_pulse_multidim(
        sim_data: options.SimulationData,
        pulse_x: torch.tensor, pulse_y: torch.tensor, grad: torch.tensor, dt_s: torch.tensor) -> torch.tensor:
    """ Calculate the propagation matrix per time step dt, with one amplitude value (complex) for the pulse
    (hard pulse approximation) and an amplitude value for the gradient"""
    # calculate rotation around x
    angle_x = 2 * torch.pi * dt_s * sim_data.gamma * pulse_x
    # calculate rotation around y
    angle_y = 2 * torch.pi * dt_s * sim_data.gamma * pulse_y
    # calculate rotation around z
    angle_z = 2 * torch.pi * dt_s * sim_data.gamma * grad * 1e-3 * sim_data.sample_axis
    r_z = rotation_matrix_z(angle_z)
    # apply all
    r_xy = torch.matmul(rotation_matrix_x(angle_x), rotation_matrix_y(angle_y))
    # r_xy dims [num_b1s steps, 4, 4]
    # r_z dims [num_samples, 4, 4]
    propagation_matrix = torch.matmul(r_z[None, :], r_xy[:, None])
    # dims [num_b1s, num_samples, 4, 4]
    relaxation_matrix = matrix_propagation_relaxation_multidim(dt_s=dt_s, sim_data=sim_data)
    # dims [num_t1s, num_t2s, 4, 4]
    # need dims [t1s, t2s, b1s, samples, 4, 4]
    return torch.matmul(propagation_matrix[None, None, :], relaxation_matrix)


def matrix_propagation_relaxation_multidim(dt_s: torch.tensor, sim_data: options.SimulationData) -> torch.tensor:
    e1 = torch.exp(-dt_s / sim_data.t1_vals)[:, None].repeat(1, sim_data.t2_vals.shape[0])
    e2 = torch.exp(-dt_s / sim_data.t2_vals)[None, :].repeat(sim_data.t1_vals.shape[0], 1)
    t_0 = torch.zeros(e1.shape, device=sim_data.device)
    t_1 = torch.ones(e1.shape, device=sim_data.device)
    relax_matrix = torch.stack([
        torch.stack([e2, t_0, t_0, t_0]),
        torch.stack([t_0, e2, t_0, t_0]),
        torch.stack([t_0, t_0, e1, t_1 - e1]),
        torch.stack([t_0, t_0, t_0, t_1])
    ])
    relax_matrix = torch.moveaxis(relax_matrix, -1, 0)
    relax_matrix = torch.moveaxis(relax_matrix, -1, 0)
    # cast to b1 and num samples dim
    return relax_matrix[:, :, None, None]


def sample_acquisition(etl_idx: int, sim_params: options.SimulationParameters, sim_data: options.SimulationData,
                       acquisition_grad: torch.tensor, dt_s: torch.tensor):
    # acquisition
    for acq_idx in range(sim_params.settings.acquisition_number):
        sim_data = propagate_gradient_pulse_relax(
            pulse_x=torch.zeros(acquisition_grad.shape[0], device=sim_data.device),
            pulse_y=torch.zeros(acquisition_grad.shape[0], device=sim_data.device),
            grad=acquisition_grad, sim_data=sim_data, dt_s=dt_s)
        # remember dims [t1s, t2s, b1s, sample, 4]
        mag_data_cmplx = torch.sum(
            sim_data.magnetization_propagation[:, :, :, :, 0], dim=-1) + 1j * torch.sum(
            sim_data.magnetization_propagation[:, :, :, :, 1], dim=-1)
        # signal tensor [t1s, t2s, b1s, etl, acq_num]
        sim_data.signal_tensor[:, :, :, etl_idx, acq_idx] = 1e3 * mag_data_cmplx * sim_params.settings.length_z / \
                                                            sim_params.settings.sample_number
    return sim_data


def sum_sample_acquisition(etl_idx: int, sim_params: options.SimulationParameters, sim_data: options.SimulationData,
                           acquisition_duration_s: torch.tensor, partial_fourier: float = 1.0):
    # timing - partial fourier assumed to be taken at beginning of echo
    dt_aq_pre = acquisition_duration_s * (partial_fourier - 1 / 2)  # cast to s
    mat_prop_relax = matrix_propagation_relaxation_multidim(dt_s=dt_aq_pre, sim_data=sim_data)
    sim_data = propagate_matrix_mag_vector(mat_prop_relax, sim_data=sim_data)
    # sum contributions
    # remember dims [t1s, t2s, b1s, sample, 4]
    mag_data_cmplx = torch.sum(
        sim_data.magnetization_propagation[:, :, :, :, 0], dim=-1) + 1j * torch.sum(
        sim_data.magnetization_propagation[:, :, :, :, 1], dim=-1)
    # emc tensor [t1s, t2s, b1s, etl]
    sim_data.emc_signal_mag[:, :, :, etl_idx] = 1e3 * torch.abs(mag_data_cmplx) * sim_params.settings.length_z / \
                                                sim_params.settings.sample_number
    sim_data.emc_signal_phase[:, :, :, etl_idx] = torch.angle(mag_data_cmplx)

    # relaxation rest of acquisition time, from mid echo til end
    dt_aq_post = acquisition_duration_s / 2
    mat_prop_relax = matrix_propagation_relaxation_multidim(dt_s=dt_aq_post, sim_data=sim_data)
    sim_data = propagate_matrix_mag_vector(mat_prop_relax, sim_data=sim_data)
    return sim_data


def generate_sample(axis: torch.tensor, extent: float):
    _sample = torch.from_numpy(stats.gennorm(24).pdf(axis / extent * 1.1) + 1e-6)
    _sample = torch.divide(_sample, torch.max(_sample))
    return _sample


if __name__ == '__main__':

    t1 = 1.5
    t2 = 0.035
    dt = 10e-6

    z_extend = 0.01  # m
    sample_num = 1200
    sample_axis = torch.linspace(-z_extend, z_extend, sample_num)
    grad_data = torch.from_numpy(np.load("../tests/grad_data.npy"))
    pulse_data = torch.from_numpy(np.load("../tests/pulse_data.npy"))

    sample = generate_sample(sample_axis, z_extend)

    magnetization = torch.zeros((sample_num, 4))
    for k in range(2):
        # set equilibrium and z start magnetization to sample values
        magnetization[:, 2 + k] = sample
    # ToDo: test case
