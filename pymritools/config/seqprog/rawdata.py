import logging
from simple_parsing import field
from pymritools.config import BaseClass
from dataclasses import dataclass
import polars as pl
import numpy as np
import pathlib as plib
import pickle

log_module = logging.getLogger(__name__)

class KTrajectory:
    def __init__(self, name: str, trajectory: np.ndarray, adc_sample_num: np.ndarray = None):
        self.name: str = name
        self.trajectory: np.ndarray = trajectory
        if adc_sample_num is None:
            log_module.debug("Assume trajectory is ordered as increasing number of sampling ADC.")
            self.adc_sample_num: np.ndarray = np.arange(trajectory.shape[0])
        else:
            self.adc_sample_num: np.ndarray = adc_sample_num

class Sample:
    def __init__(
            self,
            acquisition_type: str, scan_number: int, slice_number: int,
            phase_encode_number: int,
            echo_number: int, echo_type: str):
        self.acquisition_type: str = acquisition_type
        self.scan_number: int = scan_number
        self.slice_number: int = slice_number
        self.phase_encode_number: int = phase_encode_number
        self.echo_number: int = echo_number
        self.echo_type: str = echo_type


class Sampling:
    def __init__(self):
        self.acquisitions: list[str] = []
        self.trajectories: list[KTrajectory] = []
        self.samples: list[Sample] = []

        # self.k_trajectories: pl.DataFramepl.DataFrame().with_columns(
        #         ["acquisition", "adc_sampling_num", "k_traj_position"]
        #     )
        # self.sampling_pattern: pl.DataFrame = pl.DataFrame().with_columns(
        #     [
        #         "scan_num", "slice_num", "pe_num", "acq_type",
        #         "echo_num", "echo_type", "echo_type_num",
        #         "nav_acq"
        #     ]
        # )

    def save(self, file_name: str):
        file_name = plib.Path(file_name).absolute()
        log_module.info(f"writing file: {file_name.as_posix()}")
        if not ".pkl" in file_name.suffixes:
            log_module.info(f"Adopting file ending to .pkl")
            file_name = file_name.with_suffix(".pkl")
        if not file_name.parent.exists():
            log_module.info(f"Creating parent directory: {file_name.parent.as_posix()}")
            file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name: str):
        file_name = plib.Path(file_name).absolute()
        log_module.info(f"loading file: {file_name.as_posix()}")
        if not file_name.exists():
            err = f"File {file_name.as_posix()} does not exist!"
            log_module.error(err)
            raise FileNotFoundError(err)
        with open(file_name, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, Sampling):
            err = "Given file does not point to content representing a Sampling object!"
            log_module.error(err)
            raise TypeError(err)
        return obj

    def register_trajectory(self, trajectory: np.ndarray | list, identifier: str):
        if isinstance(trajectory, list):
            trajectory = np.array(trajectory)
        if trajectory.shape.__len__() > 1:
            adc_samples = trajectory[:, 0]
            k_read_pos = trajectory[:, 1]
        else:
            adc_samples = np.arange(trajectory.shape[0])
            k_read_pos = trajectory
        if identifier in self.acquisitions:
            err = f"trajectory identifier {identifier} exists already!"
            log_module.error(err)
            raise ValueError(err)

        self.acquisitions.append(identifier)

        found_acquisition = False
        for idx_s, s in enumerate(self.samples):
            if s.acquisition_type == identifier:
                log_module.debug("trajectory identifier was found to correspond to registered samples.")
                found_acquisition = True
        if not found_acquisition:
            log_module.debug("trajectory identifier was not found to correspond to registered samples.")

        self.trajectories.append(KTrajectory(name=identifier, trajectory=k_read_pos, adc_sample_num=adc_samples))
        # acquisition = [identifier] * trajectory.shape[0]
        #
        # df = pl.DataFrame({
        #     "acquisition": acquisition, "adc_sampling_num": adc_samples, "k_traj_position": k_read_pos
        # })
        # self.k_trajectories = pl.concat((self.k_trajectories, df))

    def register_sample(
            self, scan_num: int, slice_num: int, pe_num, echo_num: int,
            acq_type: str = "", echo_type: str = "", echo_type_num: int = -1,
            nav_acq: bool = False, nav_dir: int = 0):
        # check if entry exists already
        for idx_s, s in enumerate(self.samples):
            if scan_num == s.scan_number:
                err = f"scan number to register in sampling pattern ({scan_num}) exists already!"
                log_module.error(err)
                raise ValueError(err)
        found_trajectory = False
        for idx_t, t in enumerate(self.trajectories):
            if t.name == acq_type:
                msg = f"Sample acquisition type {acq_type} corresponds to registered trajectory already!"
                found_trajectory = True
                log_module.debug(msg)
        if not found_trajectory:
            msg = f"Sample acquisition type {acq_type} does not corresponds to any registered trajectory"
            log_module.debug(msg)
        self.samples.append(
            Sample(
                acquisition_type=acq_type, scan_number=scan_num, slice_number=slice_num,
                phase_encode_number=pe_num, echo_number=echo_num, echo_type=echo_type
            )
        )
        # build entry
        # new_row = pl.Series({
        #     "scan_num": scan_num, "slice_num": slice_num, "pe_num": pe_num, "acq_type": acq_type,
        #     "echo_num": echo_num, "echo_type": echo_type, "echo_type_num": echo_type_num,
        #     "nav_acq": nav_acq
        # })
        # # add entry
        # self.sampling_pattern = pl.concat((self.sampling_pattern, new_row.to_frame().T))

    def sampling_pattern_from_list(self, sp_list: list):
        self.samples = sp_list

    @property
    def df_sampling_pattern(self):
        return pl.DataFrame({
            "acquisition": [s.acquisition_type for s in self.samples],
            "scan_number": [s.scan_number for s in self.samples],
            "slice_number": [s.slice_number for s in self.samples],
            "phase_encode_number": [s.phase_encode_number for s in self.samples],
            "echo_number": [s.echo_number for s in self.samples],
            "echo_type": [s.echo_type for s in self.samples]
        })

    @property
    def df_trajectories(self):
        acq = []
        nums = []
        trajs = []
        for t in self.trajectories:
            acq.extend([t.name]*t.trajectory.shape[0])
            nums.extend(t.adc_sample_num.tolist())
            trajs.extend(t.trajectory.tolist())
        return pl.DataFrame({
            "acquisition": acq, "adc_sampling_num": nums, "k_traj_position": trajs
        })

    # def plot_sampling_pattern(self, output_path: typing.Union[str, plib.Path]):
    #     # ensure plib path
    #     out_path = plib.Path(output_path).absolute().joinpath("plots")
    #     out_path.mkdir(parents=True, exist_ok=True)
    #     # plot
    #     fig_nav = px.scatter(
    #         self.sampling_pattern, x=self.sampling_pattern.index, y="pe_num",
    #         color="echo_num", symbol="nav_acq",
    #         size="slice_num",
    #         labels={
    #             "index": "Scan Number", "pe_num": "# phase encode", "nav_acq": "nav",
    #             "slice_num": "# slice"
    #         }
    #     )
    #     fig_multi_acq = px.scatter(
    #         self.sampling_pattern, x=self.sampling_pattern.index, y="pe_num",
    #         color="echo_num", symbol="echo_type",
    #         size="slice_num",
    #         labels={
    #             "index": "Scan Number", "pe_num": "# phase encode", "echo_type": "echo-type",
    #             "slice_num": "# slice"
    #         }
    #     )
    #     fig_nav.update_layout(
    #         title="Sampling Pattern Sequence",
    #         xaxis_title="Number of Scan",
    #         yaxis_title="Phase Encode Line",
    #     )
    #     fig_multi_acq.update_layout(
    #         title="Sampling Pattern Sequence",
    #         xaxis_title="Number of Scan",
    #         yaxis_title="Phase Encode Line",
    #     )
    #     # save
    #     save_file = out_path.joinpath("sp_whole_pattern_nav").with_suffix(".html")
    #     log_module.info(f"\t- writing plot file: {save_file}")
    #     fig_nav.write_html(save_file)
    #     save_file = out_path.joinpath("sp_whole_pattern_acq_type").with_suffix(".html")
    #     log_module.info(f"\t- writing plot file: {save_file}")
    #     fig_multi_acq.write_html(save_file)
    #
    # def plot_k_space_trajectories(self, output_path: typing.Union[str, plib.Path]):
    #     # ensure plib path
    #     out_path = plib.Path(output_path).absolute().joinpath("plots")
    #     out_path.mkdir(parents=True, exist_ok=True)
    #     # plot
    #     fig = px.scatter(self.k_trajectories, x="adc_sampling_num", y="k_traj_position", color="acquisition")
    #     # save
    #     f_name = out_path.joinpath("k_space_trajectories").with_suffix(".html")
    #     log_module.info(f"\t\t - writing file: {f_name.as_posix()}")
    #     fig.write_html(f_name.as_posix())
    #


@dataclass
class RD(BaseClass):
    input_file: str = field(
        alias="-i", default="",
        help="Input Raw data file in .dat format."
    )
    input_sequence_config: str = field(
        alias="-seq", default="",
        help="Input Pulseq sequence configuration file."
    )
    input_sample_config: str = field(
        alias="-samp", default="",
        help="Input sampling file containing the object storing sampling pattern and k-space trajectories."
    )
    # vars & flags
    split_read_polarity: bool = field(
        alias="-spr", default=True,
        help="Split read-polarity, i.e. produce separate k_space for each readout direction. " 
             "The polarities must include 'bu' for blip up and 'bd' for bd in sample/trajectory identifiers"
    )
    remove_os: bool = field(
        alias="-rmos", default=True,
        help="Remove oversampling from kollected k-space. "
             "We might want to turn this off for some denoising procedures to extract oversampled signal lines."
    )


@dataclass
class RMOS(BaseClass):
    input_file: str = field(
        alias="-i", default="",
        help="Input Raw data file in .dat format."
    )
    data_in_kspace: bool = field(
        alias="-dk", default=True,
        help="Toggle data in k or image space."
    )
    os_factor: int = field(
        alias="-os", default=2,
        help="Set oversampling factor."
    )
    dim: int = field(
        alias="-dim", default=-1,
        help="Dimension of oversampling / read direction to reduce. "
    )
