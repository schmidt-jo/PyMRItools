import logging
import abc
import pathlib as plib

import torch

from pymritools.config.emc import EmcParameters, EmcSimSettings, SimulationData
from pymritools.config.database import DB
from pymritools.simulation.emc.core import GradPulse, SequenceTimings
from pymritools.config.rf import RFPulse
log_module = logging.getLogger(__name__)


class Simulation(abc.ABC):
    """
    Base simulation class, we want to set all parameters as variables, set up the simulation data object,
    that carries through the data iteratively, set up plotting and
    define some common functions to use for all sequence simulations.
    """
    def __init__(self, params: EmcParameters, settings: EmcSimSettings):
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
        self.settings: EmcSimSettings = settings

        # set up data, carrying through simulation
        self.data: SimulationData = SimulationData(params=self.params, settings=self.settings, device=self.device)

        # load up pulse file
        self.pulse: RFPulse = self.set_pulse()

        log_module.debug(f"\t\tSetup Plotting and Paths")
        self.fig_path: plib.Path = NotImplemented
        self._fig_magnetization_profile_snaps: list = []
        # pick middle sim range values for magnetization profile snaps
        self._fig_t1_idx: int = int(self.data.num_t1s / 2)  # t1
        self._fig_t2_idx: int = int(self.data.num_t2s / 2)  # t2
        self._fig_b1_idx: int = int(self.data.num_b1s / 2)  # b1

        # setup plotting
        if self.settings.visualize:
            # set up plotting path
            out_path = plib.Path(self.settings.out_path).absolute().joinpath("figs/")
            out_path.mkdir(parents=True, exist_ok=True)
            self.fig_path = out_path

            # set up indices for which we save the snaps
            # (otherwise we need to transfer the whole t1s * t2s * b1s * sample_num * 4 array
            # every time we save the profile)
            # choose first t1

            # save initial magnetization to snaps
            self.set_magnetization_profile_snap(snap_name="initial")

        # setup acquisition
        self.gp_acquisition = GradPulse.prep_acquisition(params=self.params)
        # give instance of sequence timings
        self.sequence_timings: SequenceTimings = SequenceTimings()
        # call specific prep
        self._prep()
        # call timing build
        self._register_sequence_timings()

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

    def set_pulse(self):
        if not self.settings.pulse_file:
            err = f"provide pulse file or set shape using the RFPulse object in 'pymritools.config.rf'"
            log_module.error(err)
            raise ValueError(err)
        path = plib.Path(self.settings.pulse_file).absolute()
        if not path.is_file():
            msg = f"could not find file {path}. Trying to insert wd"
            log_module.warning(msg)
            path = plib.Path.cwd().absolute().joinpath(self.settings.pulse_file)
            if not path.is_file():
                err = f"Could not find file {path}."
                log_module.error(err)
                raise FileNotFoundError(err)
        log_module.info(f"\t\t- loading pulse file: {path}")
        return RFPulse.load(path)

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

    def save(self):
        db = DB.from_simulation_data(params=self.params, sim_data=self.data)

        if self.settings.config_file:
            c_name = plib.Path(self.settings.config_file).absolute().stem
        else:
            c_name = "emc_config"

        save_path = plib.Path(self.settings.out_path).absolute()
        save_file = save_path.joinpath(c_name).with_suffix(".json")
        logging.info(f"Save Config File: {save_file.as_posix()}")
        self.settings.save_json(save_file.as_posix(), indent=2)

        # database
        save_file = save_path.joinpath(self.settings.database_name)
        logging.info(f"Save DB File: {save_file.as_posix()}")
        db.save(save_file)

        if self.settings.visualize:
            # plot magnetization profile snapshots
            # sim_obj.plot_magnetization_profiles(animate=False)
            # sim_obj.plot_emc_signal()
            # if settings.signal_fourier_sampling:
            #     sim_obj.plot_signal_traces()
            # plot database
            db.plot(out_path=self.fig_path)

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
