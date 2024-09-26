import dataclasses as dc
import logging

import numpy as np
import pathlib as plib
import typing
import polars as pl
import plotly.graph_objects as go
from simple_parsing.helpers import Serializable

log_module = logging.getLogger(__name__)


@dc.dataclass
class GlobalSystem:
    gamma_Hz = 42577478.518  # Hz/T
    max_grad_mT = 80.0  # mT/m

    def check_bandwidth_for_slice(self, slice_thickness_in_mm):
        max_bandwidth = self.gamma_Hz * self.max_grad_mT * 1e-3 * slice_thickness_in_mm * 1e-3
        return max_bandwidth


@dc.dataclass
class RFPulse(Serializable):
    name: str = ""
    bandwidth_in_Hz: float = 1000.0
    duration_in_us: float = 2000.0
    time_bandwidth: float = bandwidth_in_Hz * duration_in_us * 1e-6
    num_samples: int = int(duration_in_us)

    amplitude: np.ndarray = np.zeros(num_samples)
    phase: np.ndarray = np.zeros(num_samples)

    def __post_init__(self):
        # check array sizes - for some reason this is not working properly when creating class with input args
        if self.amplitude.shape[0] != self.num_samples:
            self.amplitude = np.zeros(self.num_samples)
        if self.phase.shape[0] != self.num_samples:
            self.phase = np.zeros(self.num_samples)

    def display(self):
        columns = {
            "Bandwidth": ["Hz", self.bandwidth_in_Hz],
            "Duration": ["us", self.duration_in_us],
            "Time-Bandwidth": ["1", self.time_bandwidth],
            "Number of Samples": ["1", self.num_samples]
        }
        display = pl.DataFrame(columns)
        print(display)

    def set_shape_on_raster(self, raster_time_s):
        # interpolate shape to duration raster
        N = int(self.duration_in_us * 1e-6 / raster_time_s)
        self.amplitude = np.interp(
            np.linspace(0, self.amplitude.shape[0], N),
            np.arange(self.amplitude.shape[0]),
            self.amplitude
        )
        self.phase = np.interp(
            np.linspace(0, self.phase.shape[0], N),
            np.arange(self.phase.shape[0]),
            self.phase
        )
        self.num_samples = N

    def set_flip_angle(self, flip_angle_rad: float):
        gamma_pi = GlobalSystem().gamma_Hz * 2 * np.pi
        delta_t_us = self.duration_in_us / self.num_samples
        # normalize shape
        normalized_shape = self.amplitude / np.linalg.norm(self.amplitude)
        # calculate flip angle
        flip_angle_normalized_shape = np.sum(np.abs(normalized_shape * gamma_pi)) * delta_t_us * 1e-6
        # set to new flip angle
        self.amplitude = flip_angle_rad / flip_angle_normalized_shape * normalized_shape

    @classmethod
    def load_from_txt(cls, f_name: typing.Union[str, plib.Path],
                      bandwidth_in_Hz: float = None,
                      duration_in_us: int = None,
                      time_bandwidth: float = None,
                      num_samples: int = None):
        """
        read file from .txt or .pta. Need to fill the additional specs.
        :param f_name: file name
        :param bandwidth_in_Hz: Bandwidth in Hertz (optional if duration and tbw provided)
        :param duration_in_us:  Duration in microseconds (optional if bandwidth and tbw provided)
        :param time_bandwidth: Time Bandwidth product unitless (optional if bandwidth and duration provided)
        :param num_samples: number of samples of pulse optional, if None pulse sampled per microsecond
        :return:
        """
        if bandwidth_in_Hz is None:
            if duration_in_us is None:
                err = f"No bandwidth provided: provide Duration and time-bandwidth product"
                log_module.error(err)
                raise ValueError
            bandwidth_in_Hz = time_bandwidth / duration_in_us * 1e6
        elif duration_in_us is None:
            if time_bandwidth is None:
                err = f"No duration provided: provide Bandwidth and time-bandwidth product"
                log_module.error(err)
                raise ValueError
            duration_in_us = int(1e6 * time_bandwidth / bandwidth_in_Hz)
        else:
            time_bandwidth = bandwidth_in_Hz * duration_in_us * 1e-6
        rf_cls = cls(bandwidth_in_Hz=bandwidth_in_Hz, duration_in_us=duration_in_us, time_bandwidth=time_bandwidth)
        if num_samples is None:
            num_samples = duration_in_us
        t_name = plib.Path(f_name).absolute()
        if t_name.is_file():
            # load file content
            ext_file = plib.Path(t_name)
            with open(ext_file, "r") as f:
                content = f.readlines()

            # find line where actual data starts
            start_count = -1
            while True:
                start_count += 1
                line = content[start_count]
                start_line = line.strip().split('\t')[0]
                if start_line.replace('.', '', 1).isdigit():
                    break

            # read to array
            if content.__len__() != num_samples:
                log_module.info(f"file content not matching number of samples given {num_samples}")
                num_samples = content.__len__() - start_count
                log_module.info(f"adjusting number of samples to {num_samples}")
                rf_cls.num_samples = num_samples
            content = content[start_count:start_count + num_samples]
            temp_amp = np.array([float(line.strip().split('\t')[0]) for line in content])
            temp_phase = np.array([float(line.strip().split('\t')[1]) for line in content])

            rf_cls.amplitude = temp_amp
            rf_cls.phase = temp_phase
        else:
            err = f"no file ({t_name}) found or non valid file type"
            log_module.error(err)
            raise AttributeError(err)
        if rf_cls.amplitude.shape[0] != rf_cls.phase.shape[0]:
            err = "shape of amplitude and phase do not match"
            log_module.error(err)
            raise AttributeError(err)
        if rf_cls.duration_in_us < rf_cls.num_samples:
            info = f"loaded content, i.e. number of samples {rf_cls.num_samples}, " \
                   f"set duration: {rf_cls.duration_in_us:.1f}.\n" \
                   f"-> sub us sampling. We crop the arrays to make them sampled per us ({int(rf_cls.duration_in_us)})"
            log_module.info(info)
            rf_cls.amplitude = rf_cls.amplitude[:int(rf_cls.duration_in_us)]
            rf_cls.phase = rf_cls.phase[:int(rf_cls.duration_in_us)]
            rf_cls.num_samples = int(rf_cls.duration_in_us)
        return rf_cls

    def resample_to_duration(self, duration_in_us: int):
        # want to use pulse with different duration,
        # ! in general time bandwidth properties do not have to go linearly with duration
        # we have a pulse with given tb prod at given duration,
        # if we change the duration the bandwidth is expected to change accordingly
        self.bandwidth_Hz = self.time_bandwidth / duration_in_us * 1e6
        self.duration_us = duration_in_us

    # def set_bandwidth_Hz(self, bw: float):
    #     self.bandwidth_in_Hz = bw
    #
    # def set_duration_us(self, duration: int):
    #     self.duration_in_us = duration
    #
    # def get_num_samples(self) -> int:
    #     return self.num_samples

    @property
    def dt_sampling_in_s(self) -> float:
        return 1e-6 * self.dt_sampling_in_us

    @property
    def dt_sampling_in_us(self) -> float:
        return self.duration_in_us / self.num_samples

    def plot(self, output_path: typing.Union[str, plib.Path], name: str = ""):
        if name:
            name = f"_{name}"
        # ensure plib path
        out_path = plib.Path(output_path).absolute().joinpath("plots")
        out_path.mkdir(parents=True, exist_ok=True)
        # get time steps
        dt = self.duration_in_us / self.num_samples
        # plot
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=np.arange(self.num_samples) * dt, y=self.amplitude / np.max(self.amplitude),
                name="Amplitude",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=np.arange(self.num_samples) * dt, y=self.phase / np.pi,
                name="Phase",
                mode="lines",
            )
        )
        fig.update_layout(
            title="RF Pulse",
            xaxis_title=f"Sampling point - " + '\u0394t' + f"[{int(dt)} ms]",
            yaxis_title="Norm. Amplitude | Phase [" + '\u03C0' + "]",
        )
        # save
        f_name = out_path.joinpath(f"pulse_shape{name}").with_suffix(".html")
        log_module.info(f"writing plot file: {f_name.as_posix()}")
        fig.write_html(f_name.as_posix())

