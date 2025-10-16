from dataclasses import dataclass
from simple_parsing import field

from pymritools.simulation.emc.sequence.base_sequence import SimulationData
from pymritools.simulation.emc.core import functions, GradPulse
from pymritools.config import setup_program_logging, setup_parser, BaseClass
from pymritools.config.emc import EmcSimSettings, EmcParameters
import torch
import logging
import pathlib as plib
import pickle
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

log_module = logging.getLogger(__name__)


@dataclass
class PulseSimulationSettings(BaseClass):
    pulse_name: str = field(
        alias="-pn", default="excitation",
        help="Input Pulse name to extract from the kernels file."
    )
    kernel_file: str = field(
        alias="-kf", default="",
        help="Input kernel file (.pkl) path containing the individual pulses."
    )
    sample_length: float = field(
        alias="-sl", default=0.004,
        help="Length of the slice sample to simulate."
    )
    num_samples: int = field(
        alias="-sn", default=1000,
        help="Number of isochromats to place across sample."
    )
    pulse_duration: float = field(
        alias="-pd", default=2000.0,
        help="Pulse duration in ms."
    )
    pulse_flip_angle: float = field(
        alias="-pfa", default=90.0,
        help="Pulse flip angle in degrees."
    )


class Pulse:
    def __init__(self, settings: PulseSimulationSettings):
        # load the pulse file
        kernel_file = plib.Path(settings.kernel_file)
        if not kernel_file.is_file():
            err = f"Invalid kernel file path: {kernel_file.as_posix()}"
            log_module.error(err)
            raise ValueError(err)

        with open(kernel_file, "rb") as f:
            kernels = pickle.load(f)

        k = kernels[settings.pulse_name]
        k_ref = kernels["refocus_1"]
        self.name = settings.pulse_name
        self.out_path = plib.Path(settings.out_path)
        # if self.name == "excitation":
        #     # no crushing - rephasing to calculate
        #     area_slice_select = k.grad_slice.area[1]
        #     area_rephase = - 0.5 * area_slice_select
        #     t1 = k.grad_slice.t_array_s[-2] - k.grad_slice.t_array_s[-3]
        #     t2 = k.grad_slice.t_array_s[-1] - k.grad_slice.t_array_s[-2]
        #     a = k.grad_slice.amplitude[2]
        #     b = (2 * area_rephase - a * t1) / (t1 + t2)
        #     k.grad_slice.amplitude[-2] = b
        #     k.grad_slice.area[-1] = area_rephase

        self.device = torch.device(
            f"cuda:{settings.gpu_device}" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        )

        # set some simulation parameters
        ep = EmcParameters()
        ep.sample_number = settings.num_samples
        ep.length_z = settings.sample_length
        ep.etl = 1
        ep.esp = 2
        ep.duration_excitation = settings.pulse_duration * 1e-3
        ep.duration_refocus = k_ref.rf.t_duration_s
        # set up sim data
        config = EmcSimSettings()
        config.use_gpu = settings.use_gpu
        config.visualize = settings.visualize
        config.out_path = settings.out_path
        config.pulse_file = None

        # here we fix some simple simulation parameters, see _prep
        # setup single values only
        config.t2_list = [1000]
        config.t1_list = [2]
        config.b1_list = [1.0]
        ep.etl = 1

        # changes to params, reset simulation data
        self.data = SimulationData(params=ep, settings=config, device=self.device)

        self.grad_pulse = GradPulse.prep_from_pulseq_kernel(
            kernel=k, name="excitation", device=self.device, b1s=self.data.b1_vals,
            flip_angle_rad=settings.pulse_flip_angle / 180 * torch.pi,
        )

        self.grad_pulse_ref = GradPulse.prep_from_pulseq_kernel(
            kernel=k_ref, name="refocus_1", device=self.device, b1s=self.data.b1_vals,
            flip_angle_rad=torch.pi,
        )

        if settings.visualize:
            k.plot(path=self.out_path, name=f"kernel_{settings.pulse_name}", file_suffix="html")
            self.grad_pulse.plot(b1_vals=self.data.b1_vals, fig_path=self.out_path)
            k_ref.plot(path=self.out_path, name=f"kernel_refocus_1", file_suffix="html")
            self.grad_pulse_ref.plot(b1_vals=self.data.b1_vals, fig_path=self.out_path)


    def _set_device(self):
        # set devices
        self.grad_pulse.set_device(self.device)

    def simulate(self):
        log_module.info("Simulating Pulse")

        # propagate pulse
        # pulse
        self.data = functions.propagate_gradient_pulse_relax(
            pulse_x=self.grad_pulse.data_pulse_x, pulse_y=self.grad_pulse.data_pulse_y,
            grad=self.grad_pulse.data_grad, sim_data=self.data,
            dt_s=self.grad_pulse.dt_sampling_steps_us * 1e-6
        )
        mag_exc = torch.squeeze(self.data.magnetization_propagation)
        # delay
        relax_matrix = functions.matrix_propagation_relaxation_multidim(
            dt_s=0.00182, sim_data=self.data
        )
        self.data = functions.propagate_matrix_mag_vector(relax_matrix, sim_data=self.data)

        # refocus
        self.data = functions.propagate_gradient_pulse_relax(
            pulse_x=self.grad_pulse_ref.data_pulse_x, pulse_y=self.grad_pulse_ref.data_pulse_y,
            grad=self.grad_pulse_ref.data_grad, sim_data=self.data,
            dt_s=self.grad_pulse_ref.dt_sampling_steps_us * 1e-6
        )
        self.data = functions.propagate_matrix_mag_vector(relax_matrix, sim_data=self.data)
        mag_refocus = torch.squeeze(self.data.magnetization_propagation)
        return mag_exc, mag_refocus

def plot(mag_list: list, sample_axis: torch.Tensor, out_path:plib.Path,):
    fig = psub.make_subplots(rows=2, cols=1, shared_xaxes=True)
    cmap = plc.sample_colorscale("Inferno", 2, 0.2, 0.9)
    for i, m in enumerate(mag_list):
        fig.add_trace(
            go.Scatter(
                x=sample_axis.cpu(), y=m[:, :2].norm(dim=-1).cpu().numpy(),
                name=f"Mxy", marker=dict(color=cmap[0]), showlegend=i==0
            ),
            row=1+i, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sample_axis.cpu(),
                y=m[:, 2].cpu().numpy(),
                name=f"Mz", marker=dict(color=cmap[1]), showlegend=i==0
            ),
            row=1+i, col=1,
        )
    fn = out_path.joinpath(f"mag_profiles").with_suffix(".html")
    log_module.info(f"Saving magnetization profile to {fn.as_posix()}")
    fig.write_html(fn)


def main():
    # setup logging
    setup_program_logging(name="EMC simulation - Pulse", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="EMC simulation - Pulse",
        dict_config_dataclasses={"settings": PulseSimulationSettings}
    )

    settings = PulseSimulationSettings.from_cli(args=args.settings)
    settings.display()

    try:
        pulse_sim = Pulse(settings=settings)
        mags = pulse_sim.simulate()
        plot(list(mags), pulse_sim.data.sample_axis, pulse_sim.out_path)
    except Exception as e:
        parser.print_usage()
        log_module.exception(e)
        exit(-1)


if __name__ == '__main__':
    main()
