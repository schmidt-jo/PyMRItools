import logging
import torch
import abc
from scipy import stats
from pymritools.config.emc import EmcParameters, EmcSettings
from .. import blocks
import pathlib as plib
log_module = logging.getLogger(__name__)


class Simulation(abc.ABC):
    def __init__(self, params: EmcParameters, settings: EmcSettings):
        log_module.info("__ Set-up Simulation __ ")
        # setup device
        if settings.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{settings.gpu_device}")
        else:
            device = torch.device("cpu")
        self.device: torch.device = device
        log_module.info(f"\t\t- torch device: {device}")
        # simulation variables
        self.params: EmcParameters = params
        self.settings: EmcSettings = settings
        # get sample number etc. from settings. Necessary to set acquisition sampling
        self.params.set_spatial_sample_resolution(
            sample_length_z=self.settings.length_z, sample_number_of_acq_bins=self.settings.acquisition_number
        )

        log_module.debug("\t\tSetup Simulation Data")
        # set up data, carrying through simulation
        self.sample_axis: torch.Tensor = torch.linspace(
            -self.params.length_z, self.params.length_z, self.settings.sample_number
        ).to(self.device)
        sample = torch.from_numpy(
                stats.gennorm(24).pdf(self.sample_axis / self.params.length_z * 1.1) + 1e-6
            )
        self.sample: torch.Tensor = torch.divide(sample, torch.max(sample))
        # First just define values, we set them up in the below method to be able to reuse the method
        # for different sim settings
        # t1
        self.t1_vals: torch.Tensor = NotImplemented
        self.num_t1s: int = NotImplemented
        # b1
        self.b1_vals: torch.Tensor = NotImplemented
        self.num_b1s: int = NotImplemented
        # t2
        self.t2_vals: torch.Tensor = NotImplemented
        self.num_t2s: int = NotImplemented

        self.m_init: torch.Tensor = NotImplemented

        # signal tensor is supposed to hold all acquisition points for all reads
        self.signal_tensor: torch.Tensor = NotImplemented
        # allocate
        # set emc data tensor -> dims: [t1s, t2s, b1s, ETL]
        self.signal_mag: torch.Tensor  = NotImplemented
        self.signal_phase: torch.Tensor = NotImplemented

        # set all those data up
        self.setup_simulation_data()

        log_module.debug(f"\t\tSetup Plotting and Paths")
        self.fig_path: plib.Path = NotImplemented
        self._fig_magnetization_profile_snaps: list = []
        # pick middle sim range values for magnetization profile snaps
        self._fig_t1_idx: int = int(self.num_t1s / 2)  # t1
        self._fig_t2_idx: int = int(self.num_t2s / 2)  # t2
        self._fig_b1_idx: int = int(self.num_b1s / 2)  # b1

        # setup plotting
        if self.settings.visualize:
            # set up plotting path
            out_path = plib.Path(self.settings.save_path).absolute().joinpath("figs/")
            out_path.mkdir(parents=True, exist_ok=True)
            self.fig_path = out_path

            # set up indices for which we save the snaps
            # (otherwise we need to transfer the whole t1s * t2s * b1s * sample_num * 4 array
            # every time we save the profile)
            # choose first t1

            # save initial magnetization to snaps
            self.set_magnetization_profile_snap(snap_name="initial")

        # setup acquisition
        self.gp_acquisition = blocks.GradPulse.prep_acquisition(params=self.params)
        # give instance of sequence timings
        self.sequence_timings: blocks.SequenceTimings = blocks.SequenceTimings()
        # sim info
        # set total number of simulated curves
        num_curves = self.data.t2_vals.shape[0] * self.data.b1_vals.shape[0] * self.data.t1_vals.shape[0]
        log_module.info(f"\t\t- total number of entries to simulate: {num_curves}")
        params.settings.total_num_sim = num_curves
        # call specific prep
        self._prep()
        # call timing build
        self._register_sequence_timings()

    def setup_simulation_data(self):
        # set values with some error catches
        # t1
        if isinstance(self.settings.t1_list, list):
            t1_vals = torch.tensor(self.settings.t1_list, device=self.device)
        else:
            t1_vals = torch.tensor([self.settings.t1_list], dtype=torch.float32, device=self.device)
        self.t1_vals: torch.Tensor = t1_vals
        self.num_t1s: int = self.t1_vals.shape[0]
        # b1
        self.b1_vals: torch.Tensor = self._build_array_from_list_of_lists_args(self.settings.b1_list).to(self.device)
        self.num_b1s: int = self.b1_vals.shape[0]
        # t2
        array_t2 = self._build_array_from_list_of_lists_args(self.settings.t2_list)
        array_t2 /= 1000.0  # cast to s
        self.t2_vals: torch.Tensor = array_t2.to(self.device)
        self.num_t2s: int = self.t2_vals.shape[0]

        # set magnetization vector and initialize magentization, insert axes for [t1, t2, b1] number of simulations
        m_init = torch.zeros((self.settings.sample_number, 4))
        m_init[:, 2] = self.sample
        m_init[:, 3] = self.sample
        self.m_init: torch.Tensor = m_init[None, None, None].to(self.device)

        # signal tensor is supposed to hold all acquisition points for all reads
        self.signal_tensor: torch.Tensor = torch.zeros(
            (self.num_t1s, self.num_t2s, self.num_b1s, self.params.etl, self.params.acq_number),
            dtype=torch.complex128, device=self.device
        )

        # allocate
        # set emc data tensor -> dims: [t1s, t2s, b1s, ETL]
        self.signal_mag: torch.Tensor = torch.zeros(
            (self.num_t1s, self.num_t2s, self.num_b1s, self.params.etl),
            device=self.device
        )
        self.signal_phase: torch.Tensor = torch.zeros(
            (self.num_t1s, self.num_t2s, self.num_b1s, self.params.etl),
            device=self.device
        )

    @staticmethod
    def _build_array_from_list_of_lists_args(val_list) -> torch.tensor:
        array = []
        if isinstance(val_list, list):
            for item in val_list:
                if isinstance(item, str):
                    item = [float(i) for i in item[1:-1].split(',')]
                if isinstance(item, int):
                    item = float(item)
                if isinstance(item, float):
                    array.append(item)
                else:
                    array.extend(torch.arange(*item).tolist())
        else:
            array = [val_list]
        return torch.tensor(array, dtype=torch.float32)


    @abc.abstractmethod
    def _prep(self):
        """ sequence specific preparation method (gradient pulses and init variants) """

    @abc.abstractmethod
    def _simulate(self):
        """ sequence specific simulation method"""

    @abc.abstractmethod
    def _set_device(self):
        """ sequence specific setting to put relevant tensors on device """

    @abc.abstractmethod
    def _register_sequence_timings(self):
        """ sequence specific timing objects """

    def set_magnetization_profile_snap(self, snap_name: str):
        """ add magnetization profile snapshot to list for plotting """
        t1_choice_idx = min([self.data.magnetization_propagation.shape[0] - 1, self._fig_t1_idx])
        t2_choice_idx = min([self.data.magnetization_propagation.shape[1] - 1, self._fig_t2_idx])
        b1_choice_idx = min([self.data.magnetization_propagation.shape[2] - 1, self._fig_b1_idx])
        mag_profile = self.data.magnetization_propagation[
            t1_choice_idx, t2_choice_idx, b1_choice_idx
        ].clone().detach().cpu()
        self._fig_magnetization_profile_snaps.append({
            "name": snap_name,
            "profile": mag_profile
        })

    def simulate(self):
        """ sequence specific definition of the simulation """
        # set device
        self._set_device()
        # simulate
        self._simulate()

    # def plot_magnetization_profiles(self, animate: bool = True):
    #     # pick middle sim range values
    #     b1_val = f"{self.data.b1_vals[self._fig_b1_idx].numpy(force=True):.2f}".replace(".", "p")
    #     t2_val = f"{1000*self.data.t2_vals[self._fig_t2_idx].numpy(force=True):.1f}ms".replace(".", "p")
    #     t1_val = f"{self.data.t1_vals[self._fig_t1_idx].numpy(force=True):.2f}s".replace(".", "p")
    #
    #     profiles = []
    #     dims = []
    #     names = []
    #     sample_pts = []
    #     last_name = ""
    #     for entry_idx in range(len(self._fig_magnetization_profile_snaps)):
    #         entry_dict = self._fig_magnetization_profile_snaps[entry_idx]
    #         name = entry_dict["name"]
    #         # loop to iterating characters, see if we are on same refocussing
    #         for character in name:
    #             # checking if character is numeric,
    #             # saving index
    #             if character.isdigit():
    #                 temp = name.index(character)
    #                 name = name[:temp+1]
    #                 break
    #         if name == last_name:
    #             dim_extend = "_post_acquisition"
    #         else:
    #             dim_extend = ""
    #         # on inital magnetization no different values are available
    #         mag_prof = entry_dict["profile"].numpy(force=True)
    #         profiles.extend(np.abs(mag_prof[:, 0] + 1j * mag_prof[:, 1]))
    #         dims.extend([f"abs{dim_extend}"] * mag_prof.shape[0])
    #         profiles.extend(np.angle(mag_prof[:, 0] + 1j * mag_prof[:, 1]) / np.pi)
    #         dims.extend([f"angle{dim_extend}"] * mag_prof.shape[0])
    #         profiles.extend(mag_prof[:, 2])
    #         dims.extend([f"z{dim_extend}"] * mag_prof.shape[0])
    #
    #         names.extend([name] * 3 * mag_prof.shape[0])
    #         sample_pts.extend(np.tile(self.data.sample_axis.numpy(force=True) * 1e3, 3))
    #         last_name = name
    #     df = pd.DataFrame({
    #         "profile": profiles, "dim": dims, "axis": sample_pts, "name": names
    #     })
    #     # calculate desired slice thickness from pulse & slice select
    #     bw = self.params.pulse.excitation.bandwidth_in_Hz      # Hz
    #     grad = self.params.sequence.gradient_excitation     # mT/m
    #     desired_slice_thickness_mm = np.abs(
    #         bw / self.params.sequence.gamma_hz / grad / 1e-6
    #     )
    #     plotting.plot_magnetization(
    #         mag_profile_df=df, animate=animate, name=f"t1-{t1_val}_t2-{t2_val}_b1-{b1_val}",
    #         out_path=self.fig_path, slice_thickness_mm=desired_slice_thickness_mm
    #     )
    #
    # def plot_emc_signal(self):
    #     plotting.plot_emc_sim_data(sim_data=self.data, out_path=self.fig_path)
    #
    # def plot_signal_traces(self):
    #     plotting.plot_signal_traces(sim_data=self.data, out_path=self.fig_path)