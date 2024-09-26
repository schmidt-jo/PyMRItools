#
# def single_pulse(sim_params: options.SimulationParameters):
#     """assume T2 > against pulse width"""
#     device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
#     log_module.debug(f"torch device: {device}")
#
#     # set tensor of k value-tuples to simulate for, here only b1
#     n_b1 = 1
#     # b1_vals = torch.linspace(0.5, 1.4, n_b1)
#     b1_vals = torch.tensor(1.0)
#     n_t2 = 1
#     # t2_vals_ms = torch.linspace(35, 50, n_t2)
#     t2_vals_ms = torch.tensor(50)
#
#     sim_params.settings.sample_number = 500
#     sim_params.settings.length_z = 0.005
#     sim_params.settings.t2_list = t2_vals_ms.tolist()
#     sim_params.settings.b1_list = b1_vals.tolist()
#     sim_data = options.SimulationData.from_sim_parameters(sim_params=sim_params, device=device)
#
#     grad_pulse_data = prep.GradPulse.prep_single_grad_pulse(
#         params=sim_params, excitation_flag=True, grad_rephase_factor=1.0
#     )
#     grad_pulse_data.set_device(device)
#
#     plot_idx = 0
#     fig = plotting.prep_plot_running_mag(2, 1, 0.05, 1.0)
#     # excite only
#     fig = plotting.plot_running_mag(fig, sim_data=sim_data, id=plot_idx)
#     plot_idx += 1
#
#     # --- starting sim matrix propagation --- #
#     log_module.debug("excitation")
#     sim_data = functions.propagate_gradient_pulse_relax(
#         grad=grad_pulse_data.data_grad, pulse_x=grad_pulse_data.data_pulse_x,
#         pulse_y=grad_pulse_data.data_pulse_y, dt_s=grad_pulse_data.dt_sampling_steps * 1e-6, sim_data=sim_data)
#
#     fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
#     plot_idx += 1
#     plotting.display_running_plot(fig)