"""
event specific classes

"""
import types
import typing

import matplotlib.pyplot as plt
import pypulseq as pp
from pymritools.config.rf import RFPulse
import numpy as np
import logging
import copy

log_module = logging.getLogger(__name__)


class Event:
    def __init__(self):
        self.t_duration_s: float = 0.0
        self.system: pp.Opts = NotImplemented
        self.t_mid: float = 0.0
        # can attribute the timing of the middle event for using gradients and calculate timing.

    def get_duration(self):
        raise NotImplementedError

    def to_simple_ns(self):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)


class RF(Event):
    def __init__(self):
        super().__init__()
        self.flip_angle_deg: typing.Union[float, np.ndarray] = 0.0
        self.phase_deg: typing.Union[float, np.ndarray] = 0.0
        self.pulse_type: str = "exc"
        self.extRfFile: str = ""

        self.flip_angle_rad: typing.Union[float, np.ndarray] = self.flip_angle_deg / 180.0 * np.pi
        self.phase_rad: typing.Union[float, np.ndarray] = self.phase_deg / 180.0 * np.pi

        self.freq_offset_hz: float = 0.0
        self.phase_offset_rad: float = 0.0

        self.t_delay_s: float = 0.0
        self.t_ringdown_s: float = 0.0
        self.t_dead_time_s: float = 0.0
        self.t_array_s: np.ndarray = np.zeros(0)

        self.bandwidth_hz: float = 0.0
        self.time_bandwidth: float = 0.0

        self.signal: np.ndarray = np.zeros(0, dtype=complex)

    @classmethod
    def load_from_pypsi_pulse(cls, fname: str, flip_angle_rad: float, phase_rad: float, system: pp.Opts,
                              duration_s: float = 2e-3, delay_s: float = 0.0, pulse_type: str = 'excitation'):
        rf_instance = cls()
        rf_instance.system = system
        rf = RFPulse.load(fname)
        rf_instance.extRfFile = fname
        rf_instance.pulse_type = pulse_type

        # get signal envelope
        signal = rf.amplitude * np.exp(1j * rf.phase)
        # calculate raster with assigned duration, we set the signal to be rastered on rf raster of 1 us
        delta_t = system.rf_raster_time
        t_array_s = rf_instance.set_on_raster(np.arange(0, int(duration_s * 1e6)) * 1e-6)
        # interpolate signal to new time
        signal_interp = np.interp(
            t_array_s,
            xp=np.linspace(0, duration_s, rf.num_samples),
            fp=signal
        )
        # make sure to start and end with 0
        signal_interp[0] = 0
        signal_interp[-1] = 0

        # normalise flip angle
        flip = np.sum(np.abs(signal_interp)) * delta_t * 2 * np.pi

        # assign values
        rf_instance.signal = signal_interp * flip_angle_rad / flip
        rf_instance.flip_angle_rad = flip_angle_rad
        rf_instance.flip_angle_deg = flip_angle_rad / np.pi * 180.0

        rf_instance.phase_rad = phase_rad
        rf_instance.phase_deg = phase_rad / np.pi * 180.0

        rf_instance.t_duration_s = duration_s
        rf_instance.time_bandwidth = rf.time_bandwidth
        rf_instance.bandwidth_hz = rf_instance.time_bandwidth / duration_s
        rf_instance.t_array_s = t_array_s

        rf_instance.t_delay_s = delay_s
        rf_instance.t_ringdown_s = system.rf_ringdown_time
        rf_instance.t_dead_time_s = system.rf_dead_time
        rf_instance._set_central_time()
        return rf_instance

    @classmethod
    def make_sinc_pulse(cls, flip_angle_rad: float, system: pp.Opts, phase_rad: float = 0.0,
                        pulse_type: str = 'excitation',
                        delay_s: float = 0.0, duration_s: float = 2e-3,
                        freq_offset_hz: float = 0.0, phase_offset_rad: float = 0.0,
                        time_bw_prod: float = 2):
        rf_instance = cls()
        rf_instance.system = system
        rf_simple_ns = pp.make_sinc_pulse(
            use=pulse_type,
            flip_angle=flip_angle_rad,
            delay=delay_s,
            duration=duration_s,
            freq_offset=freq_offset_hz,
            phase_offset=phase_offset_rad + phase_rad,
            return_gz=False,
            time_bw_product=time_bw_prod,
            system=system
        )
        rf_instance.flip_angle_deg = flip_angle_rad * 180.0 / np.pi
        rf_instance.phase_deg = phase_offset_rad * 180.0 / np.pi
        rf_instance.pulse_type = pulse_type
        rf_instance.extRfFile = ""

        rf_instance.flip_angle_rad = flip_angle_rad
        rf_instance.phase_rad = phase_rad

        rf_instance.freq_offset_hz = freq_offset_hz
        rf_instance.phase_offset_rad = phase_offset_rad

        rf_instance.t_delay_s = delay_s
        rf_instance.t_duration_s = duration_s
        rf_instance.t_ringdown_s = system.rf_ringdown_time
        rf_instance.t_dead_time_s = system.rf_dead_time
        rf_instance.t_array_s = rf_instance.set_on_raster(np.linspace(0, duration_s, rf_simple_ns.signal.shape[0]))

        rf_instance.bandwidth_hz = time_bw_prod / duration_s
        rf_instance.time_bandwidth = time_bw_prod

        rf_instance.signal = rf_simple_ns.signal
        rf_instance.system = system
        rf_instance._set_central_time()
        return rf_instance

    @classmethod
    def make_gauss_pulse(cls, flip_angle_rad: float, system: pp.Opts, phase_rad: float = 0.0,
                         pulse_type: str = 'excitation',
                         delay_s: float = 0.0, duration_s: float = 2e-3,
                         freq_offset_hz: float = 0.0, phase_offset_rad: float = 0.0,
                         time_bw_prod: float = 2):
        rf_instance = cls()
        rf_instance.system = system
        rf_simple_ns = pp.make_gauss_pulse(
            use=pulse_type,
            flip_angle=flip_angle_rad,
            delay=delay_s,
            duration=duration_s,
            freq_offset=freq_offset_hz,
            phase_offset=phase_offset_rad + phase_rad,
            return_gz=False,
            time_bw_product=time_bw_prod,
            system=system
        )
        rf_instance.flip_angle_deg = flip_angle_rad * 180.0 / np.pi
        rf_instance.phase_deg = phase_offset_rad * 180.0 / np.pi
        rf_instance.pulse_type = pulse_type
        rf_instance.extRfFile = ""

        rf_instance.flip_angle_rad = flip_angle_rad
        rf_instance.phase_rad = phase_rad

        rf_instance.freq_offset_hz = freq_offset_hz
        rf_instance.phase_offset_rad = phase_offset_rad

        rf_instance.t_delay_s = delay_s
        rf_instance.t_duration_s = duration_s
        rf_instance.t_ringdown_s = system.rf_ringdown_time
        rf_instance.t_dead_time_s = system.rf_dead_time
        rf_instance.t_array_s = rf_instance.set_on_raster(np.linspace(0, duration_s, rf_simple_ns.signal.shape[0]))

        rf_instance.bandwidth_hz = time_bw_prod / duration_s
        rf_instance.time_bandwidth = time_bw_prod

        rf_instance.signal = rf_simple_ns.signal
        rf_instance.system = system
        rf_instance._set_central_time()
        return rf_instance

    def get_duration(self):
        return self.t_delay_s + self.t_duration_s + self.t_ringdown_s

    def set_on_raster(self, input_value: typing.Union[int, float, np.ndarray]):
        is_single = isinstance(input_value, (int, float))
        if is_single:
            us_value = np.array(input_value) * 1e6
        else:
            us_value = 1e6 * input_value
        us_raster = 1e6 * self.system.rf_raster_time
        choice = us_value % us_raster
        us_value[choice < 1e-4] = us_value[choice < 1e-4]
        us_value[choice > 1e-4] = np.round(us_value[choice > 1e-4] / us_raster) * us_raster
        if is_single:
            return 1e-6 * us_value[0]
        else:
            return 1e-6 * us_value

    def to_simple_ns(self):
        return types.SimpleNamespace(
            use=self.pulse_type, dead_time=self.t_dead_time_s, delay=self.t_delay_s,
            freq_offset=self.freq_offset_hz, phase_offset=self.phase_offset_rad,
            ringdown_time=self.t_ringdown_s, shape_dur=self.t_duration_s,
            signal=self.signal, t=self.t_array_s,
            type='rf'
        )

    def _set_central_time(self):
        """
        calculate the central point of rf shape
        """
        # Detect the excitation peak; if i is a plateau take its center
        rf_max = np.max(np.abs(self.signal))
        i_peak = np.where(np.abs(self.signal) >= rf_max - 1e-9)[0]
        self.t_mid = (self.t_array_s[i_peak[0]] + self.t_array_s[i_peak[-1]]) / 2

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        amplitude = np.abs(self.signal)
        phase = np.angle(self.signal) + self.phase_rad
        ax.plot(np.linspace(0, self.t_duration_s, self.signal.shape[0]), amplitude, label='amp')
        ax.plot(np.linspace(0, self.t_duration_s, self.signal.shape[0]), phase, label='phase')
        ax.legend()
        plt.show()


class GRAD(Event):
    def __init__(self):
        super().__init__()
        self.channel: str = 'z'
        self.amplitude: typing.Union[float, np.ndarray] = 0.0
        self.area: typing.Union[float, np.ndarray] = 0.0
        self.flat_area: float = 0.0

        self.t_array_s: np.ndarray = np.zeros(0)
        self.t_delay_s: float = 0.0
        self.t_fall_time_s = 1e-5
        self.t_rise_time_s = 1e-5
        self.t_flat_time_s = 1e-5

        self.system: pp.Opts = pp.Opts()

        self.max_slew: float = self.system.max_slew
        self.max_grad: float = self.system.max_grad

        # needed in this project for referencing slice select extended gradients. easy hack
        self.slice_select_amplitude: float = NotImplemented
        self.slice_select_duration: float = NotImplemented

    def set_on_raster(self, value: float, return_delay: bool = False, double: bool = True):
        raster_time = float(self.system.grad_raster_time)
        if double:
            # helps with maintaining raster when calculating esp
            raster_time *= 2.0
        if np.abs(value) % raster_time < 1e-9:
            rastered_value = value
        else:
            rastered_value = np.ceil(value / raster_time) * raster_time
        if not return_delay:
            return rastered_value
        else:
            delay = (rastered_value - value) / 2
            return rastered_value, delay

    @classmethod
    def make_trapezoid(cls, channel: str, system: pp.Opts, amplitude: float = 0.0, area: float = None,
                       delay_s: float = 0.0, duration_s: float = 0.0,
                       flat_area: float = 0.0, flat_time: float = -1.0,
                       ramp_time: float = 0.0):
        grad_instance = cls()
        grad_instance.system = system
        # some timing checks
        if flat_time > 1e-7:
            flat_time_set = flat_time
            flat_time = grad_instance.set_on_raster(flat_time, double=False)
            if np.abs(flat_time_set - flat_time) > 1e-9:
                # if the amplitude is given, the gradient moment across the original set flat time remains equal, if
                # the flat time needs to be adjusted / prolonged due to raster. the flat area increases though.
                # if the flat area is given, the amplitude is calculated via: flat_area / flat_time, if the time increased
                # due to rastering, we get a smaller amplitude, leading to an altered area across the
                # originally set flat time.
                # this might lead to deviations in readout scenarios. where the flat time and area
                # might stem from adc timings. or would lead to minute slice select thickness changes.
                # if we want to keep this flat area across the set unrastered flat time equal
                # and just append additional area, we need to increase the area by the same relative amount
                if np.abs(flat_area) > 1e-9:
                    flat_area *= flat_time / flat_time_set
        if duration_s > 1e-7:
            duration_s = grad_instance.set_on_raster(duration_s, double=False)
        if ramp_time > 1e-7:
            ramp_time = grad_instance.set_on_raster(ramp_time, double=False)

        grad_simple_ns = pp.make_trapezoid(
            channel=channel,
            amplitude=amplitude,
            area=area,
            delay=delay_s,
            duration=duration_s,
            flat_area=flat_area,
            flat_time=flat_time,
            rise_time=ramp_time,
            fall_time=ramp_time,
            system=system
        )
        grad_instance.channel = channel
        grad_instance.amplitude = np.array([
            0.0, grad_simple_ns.amplitude,
            grad_simple_ns.amplitude, 0.0
        ])
        grad_instance.area = grad_simple_ns.area
        grad_instance.flat_area = grad_simple_ns.flat_area

        grad_instance.t_array_s = np.array([
            0.0, grad_simple_ns.rise_time,
            grad_simple_ns.rise_time + grad_simple_ns.flat_time,
            grad_simple_ns.rise_time + grad_simple_ns.flat_time + grad_simple_ns.fall_time
        ])
        grad_instance.t_delay_s = delay_s
        grad_instance.t_fall_time_s = grad_simple_ns.rise_time
        grad_instance.t_rise_time_s = grad_simple_ns.rise_time
        grad_instance.t_flat_time_s = grad_simple_ns.flat_time

        grad_instance.system = system

        grad_instance.max_slew = system.max_slew
        grad_instance.max_grad = system.max_grad
        grad_instance.t_duration_s = grad_instance.get_duration()
        grad_instance.t_mid = grad_instance.get_duration() / 2
        return grad_instance

    @classmethod
    def make_slice_selective(cls, pulse_bandwidth_hz: float, slice_thickness_m: float, duration_s: float,
                             system: pp.Opts, pre_moment: float = 0.0, re_spoil_moment: float = 0.0,
                             rephase: float = 0.0, t_minimum_re_grad: float = 0.0, adjust_ramp_area: float = 0.0):
        """
        create slice selective gradient with merged pre and re moments (optional)
        - one can set the minimum time for those symmetrical moments with t_minimum_re_grad to match other grad timings
        - the rephase parameter gives control over rephasing the slice select moment in the re-moment.
            It can be helpful to introduce correction factors: rephase = 1.0 will do
            conventional rephasing of half of the slice select area.
            From simulations it was found that eg. for one particular used slr pulse an increase of 8%
            gives better phase properties. hence one could use rephase=1.08.
        - adjust_ramp_area gives control over additional adjustments.
            This is used to account for the ramp area of a successive slice select pulse like refocusing
            to merge gradients.
        - ensure arbitrary pulse-grad shape obeys rf dead time (rf cannot start before dead time delay)
        """
        # init
        grad_instance = cls()
        grad_instance.system = system
        duration_s, rf_raster_delay = grad_instance.set_on_raster(duration_s, return_delay=True)
        # set slice select amplitude
        amplitude = pulse_bandwidth_hz / slice_thickness_m
        amps = [0.0]
        times = [0.0]
        areas = []
        # set minimum time on raster
        t_minimum_re_grad = grad_instance.set_on_raster(t_minimum_re_grad)

        # pre moment
        if np.abs(pre_moment) > 1e-7:
            # add to area array
            areas.append(pre_moment)
            # we assume moment of slice select and pre phaser to act in same direction
            if np.sign(pre_moment) != np.sign(amplitude):
                log_module.error(f"pre-phase / spoil pre -- slice select not optimized for opposite sign grads")
                # we can adopt here also with doubling the ramp times in case we have opposite signs

            # (1) check if ramp to amplitude exceeds set area if using maximum slew, i.e. smallest possible area
            grad_instance._calc_check_single_ramp_area_vs_set_area(
                area=pre_moment, amplitude=amplitude, identifier="pre-phasing / pre-spoil"
            )

            # (2) check if max grad and max slew already suffice in setting area
            pre_grad_amplitude, t_flat = grad_instance._calc_check_double_ramp_area_vs_set_area(
                area=pre_moment, amplitude=amplitude, identifier="pre-phasing / pre-spoil"
            )

            # (3) if we set times on gradient raster we might need to adopt gradient values
            pre_grad_amplitude, t_ru, t_rd, t_flat = grad_instance._reset_amplitude_calc_ramps(
                area=pre_moment, amplitude_to_set=pre_grad_amplitude, amplitude_fixed=amplitude, t_flat=t_flat
            )

            # (4) check if we have a requirement for the minimal time, if so we can possibly relax gradient stress
            duration_pre_grad = t_ru + t_rd + t_flat
            if duration_pre_grad < t_minimum_re_grad:
                # stretch to minimum required time if we have such
                # update amplitude
                b = - system.max_slew * t_minimum_re_grad + amplitude
                c = amplitude ** 2 + system.max_slew * pre_moment
                pre_grad_amplitude = -b / 2 + max(
                    np.sqrt(b ** 2 - 4 * c), - np.sqrt(b ** 2 - 4 * c)
                ) / 2
                t_ru = grad_instance.set_on_raster(np.abs(pre_grad_amplitude) / system.max_slew)
                t_rd = grad_instance.set_on_raster(np.abs(pre_grad_amplitude - amplitude) / system.max_grad)
                t_flat = t_minimum_re_grad - t_ru - t_rd
                pre_grad_amplitude, t_ru, t_rd, t_flat = grad_instance._reset_amplitude_calc_ramps(
                    area=pre_moment, amplitude_to_set=pre_grad_amplitude, amplitude_fixed=amplitude, t_flat=t_flat
                )

            # (5) build gradient shape for prephasing grad
            times.append(times[-1] + t_ru)
            amps.append(pre_grad_amplitude)
            if t_flat > 1e-7:
                amps.append(pre_grad_amplitude)
                times.append(times[-1] + t_flat)
            times.append(times[-1] + t_rd)
            amps.append(amplitude)
        else:
            # ramp up only, no pre moment, ensure at least rf dead time
            duration_pre_grad = max(
                grad_instance.set_on_raster(np.abs(amplitude / system.max_slew)),
                grad_instance.set_on_raster(system.rf_dead_time)
            )
            times.append(times[-1] + duration_pre_grad)
            amps.append(amplitude)
            areas.append(np.trapezoid(x=times, y=amps))

        # flat part of slice select gradient. an rf would start here, hence save delay
        delay = times[-1] + rf_raster_delay
        amps.append(amplitude)
        times.append(times[-1] + duration_s)
        areas.append(amplitude * duration_s)
        re_start_time = times[-1]

        # re / spoil moment
        # if np.abs(re_spoil_moment) > 1e-10: always build re-gradient to rephase slice selection,
        # either with or without spoiling
        # if np.sign(re_spoil_moment) != np.sign(amplitude):
        #     log_module.error(f"pre-phase / spoil pre -- slice select not optimized for opposite sign grads")
            # we can adopt here also with doubling the ramp times in case we have opposite signs
        re_spoil_moment += - 0.5 * rephase * areas[-1]
        # very specific requirement jstmc sequence. adjust for ramp up of next slice selective gradient
        re_spoil_moment -= adjust_ramp_area
        areas.append(re_spoil_moment)
        # (1) check if ramp to amplitude exceeds set area if using maximum slew, i.e. smallest possible area
        grad_instance._calc_check_single_ramp_area_vs_set_area(
            area=re_spoil_moment, amplitude=amplitude, identifier="re-phasing / spoil"
        )

        # (2) check if max grad and max slew already suffice in setting area
        re_grad_amplitude, t_flat = grad_instance._calc_check_double_ramp_area_vs_set_area(
            area=re_spoil_moment, amplitude=amplitude, identifier="re-phasing / spoil"
        )

        # (3) if we set times on gradient raster we might need to adopt gradient values
        re_grad_amplitude, t_ru, t_rd, t_flat = grad_instance._reset_amplitude_calc_ramps(
            area=re_spoil_moment, amplitude_to_set=re_grad_amplitude, amplitude_fixed=amplitude, t_flat=t_flat
        )

        # (4) check if we have a requirement for the minimal time, if so we can possibly relax gradient stress
        duration_re_grad = t_ru + t_rd + t_flat
        if duration_re_grad < t_minimum_re_grad:
            # stretch to minimum required time if we have such
            # update amplitude
            b = - system.max_slew * t_minimum_re_grad + amplitude
            c = amplitude ** 2 + system.max_slew * re_spoil_moment
            re_grad_amplitude = -b / 2 + max(
                np.sqrt(b ** 2 - 4 * c), - np.sqrt(b ** 2 - 4 * c)
            ) / 2
            t_ru = grad_instance.set_on_raster(np.abs(re_grad_amplitude) / system.max_slew)
            t_rd = grad_instance.set_on_raster(np.abs(re_grad_amplitude - amplitude) / system.max_grad)
            t_flat = t_minimum_re_grad - t_ru - t_rd
            re_grad_amplitude, t_ru, t_rd, t_flat = grad_instance._reset_amplitude_calc_ramps(
                area=re_spoil_moment, amplitude_to_set=re_grad_amplitude,
                amplitude_fixed=amplitude, t_flat=t_flat
            )

        # (5) build gradient shape for re/spoil grad
        times.append(times[-1] + t_rd)
        amps.append(re_grad_amplitude)
        if t_flat > 1e-7:
            amps.append(re_grad_amplitude)
            times.append(times[-1] + t_flat)
        times.append(times[-1] + t_ru)
        amps.append(0.0)
        # else:
        #     # ramp down only, no re moment
        #     duration_re_grad = grad_instance.set_on_raster(np.abs(amplitude / system.max_slew))
        #     times.append(times[-1] + duration_re_grad)
        #     amps.append(0.0)

        # end of re gradient
        re_end_time = times[-1]
        duration_re_grad = float(re_end_time - re_start_time)
        amps = np.array(amps)
        times = np.array(times)

        times = np.array(times)
        grad_instance.channel = 'z'
        grad_instance.amplitude = np.array(amps)
        grad_instance.area = np.array(areas)
        grad_instance.flat_area = np.zeros(3)

        grad_instance.t_array_s = times
        grad_instance.t_delay_s = 0.0
        grad_instance.t_fall_time_s = times[-1] - times[-2]
        grad_instance.t_rise_time_s = times[1] - times[0]
        grad_instance.t_flat_time_s = 0.0

        # for referencing slice selective part:
        grad_instance.slice_select_duration = duration_s
        grad_instance.slice_select_amplitude = amplitude

        grad_instance.system = system

        grad_instance.max_slew = system.max_slew
        grad_instance.max_grad = system.max_grad
        grad_instance.t_duration_s = grad_instance.get_duration()
        # last sanity check max grad / slew times
        if np.max(np.abs(amps)) > system.max_grad:
            err = f"amplitude violation, maximum gradient exceeded"
            log_module.error(err)
            raise ValueError(err)
        grad_slew = np.abs(np.diff(amps) / np.diff(times))
        if np.max(grad_slew) > system.max_slew:
            err = f"slew rate violation, maximum slew rate exceeded"
            log_module.error(err)
            raise ValueError(err)

        return grad_instance, delay, duration_re_grad

    def _calc_check_single_ramp_area_vs_set_area(self, area: float, amplitude: float, identifier: str = ""):
        a = amplitude ** 2 / 2 / self.system.max_slew
        if np.abs(a) > np.abs(area) + 1e-5:
            msg = (
                "area cannot accommodated with maximum slew. "
                "Area through ramps already exceeds set value. "
                f"choose higher {identifier} moment"
            )
            log_module.error(msg)
            raise ValueError(msg)

    def _calc_check_double_ramp_area_vs_set_area(self, area: float, amplitude: float, identifier: str = ""):
        """
        calculate and check if area produced by max ramps already suffices to cover set area
        """
        # calculate area of two ramps from 0 to max amp to set amp.
        a = (self.system.max_grad ** 2 - amplitude ** 2 / 2) / self.system.max_slew
        # check if this area is bigger than set area
        if np.abs(a) > np.abs(area) + 1e-5:
            # ramp parts of gradient already suffice to cover area, dont need flat gradient.
            # need to update grad value to get correct area
            grad_amplitude = np.sign(area) * np.sqrt(
                np.abs(area) * self.system.max_slew + np.abs(amplitude) ** 2 / 2
            )
            if np.abs(grad_amplitude) < np.abs(amplitude):
                err = f"area cannot accommodated with maximum slew. Choose higher {identifier} moment."
                log_module.error(err)
                raise ValueError(err)
            t_flat = 0.0
        else:
            # fix gradient at max value and calculate flat time
            grad_amplitude = self.system.max_grad
            t_flat = self.set_on_raster((np.abs(area) - np.abs(a)) / np.abs(grad_amplitude))
        return grad_amplitude, t_flat

    def _reset_amplitude_calc_ramps(
            self, area: float, amplitude_to_set: float, amplitude_fixed: float, t_flat: float = 0):
        t_ru = self.set_on_raster(np.abs(amplitude_to_set) / self.system.max_slew)
        t_rd = self.set_on_raster(np.abs(amplitude_to_set - amplitude_fixed) / self.system.max_slew)
        t_flat = self.set_on_raster(t_flat)
        a = area - 0.5 * t_rd * amplitude_fixed
        b = 0.5 * t_ru + t_flat + 0.5 * t_rd
        return a / b, t_ru, t_rd, t_flat

    @classmethod
    def sym_grad(cls, system: pp.Opts, channel: str = 'x', pre_delay: float = 0.0, area_lobe: float = 0.0,
                 amplitude_lobe: float = 0.0, duration_lobe: float = 0.0, duration_between: float = 0.0,
                 reverse_second_lobe: bool = False):
        grad_instance = cls()
        grad_instance.system = system
        duration_lobe = grad_instance.set_on_raster(float(duration_lobe))
        duration_between = grad_instance.set_on_raster(duration_between)

        grad_ns = pp.make_trapezoid(
            channel=channel, area=area_lobe, amplitude=amplitude_lobe, duration=duration_lobe, system=system
        )

        # set up arrays
        grad_instance.t_delay_s = pre_delay
        times = np.array([
            0.0,
            grad_ns.rise_time,
            grad_ns.rise_time + grad_ns.flat_time,
            duration_lobe,
            duration_lobe + duration_between,
            duration_lobe + duration_between + grad_ns.rise_time,
            duration_lobe + duration_between + grad_ns.rise_time + grad_ns.flat_time,
            2 * duration_lobe + duration_between,
        ])
        # for second lobe
        sign = 1
        if reverse_second_lobe:
            sign *= -1
        amps = np.array([
            0.0,
            grad_ns.amplitude,
            grad_ns.amplitude,
            0.0,
            0.0,
            sign * grad_ns.amplitude,
            sign * grad_ns.amplitude,
            0.0
        ])

        areas = [grad_ns.area, 0, sign * grad_ns.area]

        grad_instance.channel = channel
        grad_instance.amplitude = np.array(amps)
        grad_instance.area = np.array(areas)

        grad_instance.t_array_s = times
        grad_instance.t_fall_time_s = times[-1] - times[-2]
        grad_instance.t_rise_time_s = times[1] - times[0]

        grad_instance.system = system

        grad_instance.max_slew = system.max_slew
        grad_instance.max_grad = system.max_grad
        grad_instance.t_duration_s = grad_instance.get_duration()
        # last sanity check max grad / slew times
        if np.max(np.abs(amps)) > system.max_grad:
            err = f"amplitude violation, maximum gradient exceeded"
            log_module.error(err)
            raise ValueError(err)
        grad_slew = np.abs(np.diff(amps) / np.diff(times))
        if np.max(grad_slew) > system.max_slew:
            err = f"slew rate violation, maximum slew rate exceeded"
            log_module.error(err)
            raise ValueError(err)
        return grad_instance

    def get_duration(self):
        # 0 for empty init grad
        t = self.t_delay_s
        if self.t_array_s.__len__() < 1:
            return t
        return t + self.t_array_s[-1]

    def to_simple_ns(self):
        return types.SimpleNamespace(
            channel=self.channel, type='grad',
            delay=self.t_delay_s, first=self.amplitude[0], last=self.amplitude[-1],
            shape_dur=self.t_duration_s, tt=self.t_array_s, waveform=self.amplitude
        )

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        gamma = self.system.gamma
        amplitude = self.amplitude / gamma * 1e3
        ax.plot(self.t_array_s * 1e3, amplitude)
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("grad amp [mT/m]")
        plt.show()

    def calculate_asym_area(self, forward: bool = True):
        amps = self.amplitude
        times = self.t_array_s
        if not forward:
            amps = amps[::-1]
            times = times[::-1]
        amplitude_a = amps[1]
        amplitude_b = amps[3]
        delta_t = np.abs(np.diff(times))[:3]
        factor = np.array([0.5, 1.0, 0.5])
        area = amplitude_a * np.sum(factor * delta_t) + amplitude_b * delta_t[2] / 2
        return area


class ADC(Event):
    def __init__(self):
        super().__init__()
        self.num_samples: int = 0

        self.t_dwell_s: float = 0.0
        self.t_delay_s: float = 0.0
        self.t_duration_s: float = 0.0
        self.t_dead_time_s: float = 0.0

        self.freq_offset_hz: float = 0.0
        self.phase_offset_rad: float = 0.0

        self.system: pp.Opts = pp.Opts()

    @classmethod
    def make_adc(cls, system: pp.Opts,
                 num_samples: int = 0, delay_s: float = 0, duration_s: float = 0,
                 dwell: float = 0, freq_offset_hz: float = 0, phase_offset_rad: float = 0.0
                 ):
        adc_ns = pp.make_adc(
            num_samples=num_samples,
            delay=delay_s,
            duration=duration_s,
            dwell=dwell,
            freq_offset=freq_offset_hz,
            phase_offset=phase_offset_rad,
            system=system
        )
        adc_instance = cls()
        adc_instance.system = system
        adc_instance.num_samples = adc_ns.num_samples
        adc_instance.t_delay_s = adc_ns.delay
        adc_instance.t_dwell_s = adc_ns.dwell
        adc_instance.t_duration_s = adc_ns.dwell * adc_ns.num_samples
        adc_instance.t_dead_time_s = adc_ns.dead_time
        adc_instance.freq_offset_hz = adc_ns.freq_offset
        adc_instance.phase_offset_rad = adc_ns.phase_offset
        adc_instance.set_on_raster()
        return adc_instance

    def get_duration(self):
        return self.t_duration_s + self.t_delay_s + self.t_dead_time_s

    def to_simple_ns(self):
        return types.SimpleNamespace(
            dead_time=self.t_dead_time_s, delay=self.t_delay_s, dwell=self.t_dwell_s,
            freq_offset=self.freq_offset_hz, num_samples=self.num_samples,
            phase_offset=self.phase_offset_rad, type='adc'
        )

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        amplitude = np.array([0, 0, 1, 1, 0])
        times = np.array([
            0,
            self.t_delay_s - 1e-9,
            self.t_delay_s + 1e-9,
            (self.t_duration_s + self.t_delay_s) - 1e-9,
            (self.t_duration_s + self.t_delay_s) + 1e-9
        ])
        amp_dead_time = [0, 1, 1, 0]
        t_dead_time = [
            (self.t_duration_s + self.t_delay_s) - 1e-9,
            (self.t_duration_s + self.t_delay_s) + 1e-9,
            self.get_duration() - 1e-9,
            self.get_duration() + 1e-9,
        ]
        ax.plot(times * 1e3, amplitude)
        ax.fill_between(amp_dead_time, t_dead_time)
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("ADC")
        plt.show()

    def set_on_raster(self):
        raster_time = float(self.system.adc_raster_time)
        to_set = ["t_dwell_s", "t_delay_s", "t_duration_s"]
        # save number of samples
        n = int(self.t_duration_s / self.t_dwell_s)
        for key in to_set:
            value = self.__getattribute__(key)
            if np.abs(value) % raster_time > 1e-9:
                rastered_value = np.ceil(value / raster_time) * raster_time
                self.__setattr__(key, rastered_value)
                if key == "t_dwell_s":
                    # if dwell changed adopt
                    self.t_duration_s = n * self.t_dwell_s


class DELAY(Event):
    def __init__(self):
        super().__init__()
        self.system = pp.Opts()

    @classmethod
    def make_delay(cls, delay_s: float, system: pp.Opts = pp.Opts()):
        delay_instance = cls()
        delay_instance.system = system
        delay_instance.t_duration_s = delay_s
        delay_instance.set_on_block_raster()
        return delay_instance

    def check_on_block_raster(self) -> bool:
        # get raster in us
        us_raster = 1e6 * self.system.grad_raster_time
        # get delay value in us
        us_value = 1e6 * self.t_duration_s
        # check the modulo of he value against the raster, if its below a threshold
        # we dont need to do something about the value, otherwise raster
        if us_value % us_raster > 1e-4:
            # we need to put the value on the raster
            us_value = np.round(us_value / us_raster) * us_raster
        # cast back to seconds
        rastered_value = 1e-6 * us_value
        if np.abs(rastered_value - self.t_duration_s) > 1e-8:
            return False
        else:
            return True

    def set_on_block_raster(self):
        us_raster = 1e6 * self.system.grad_raster_time
        us_value = 1e6 * self.t_duration_s
        if us_value % us_raster < 1e-4:
            rastered_value = us_value
        else:
            rastered_value = int(np.ceil(us_value / us_raster) * us_raster)
        self.t_duration_s = rastered_value * 1e-6

    def get_duration(self):
        return self.t_duration_s

    def to_simple_ns(self):
        return types.SimpleNamespace(delay=self.t_duration_s, type='delay')


if __name__ == '__main__':
    rf = pp.make_sinc_pulse(
        np.pi / 2,
        system=pp.Opts(),
        use='excitation'
    )
    rf_new = RF.make_sinc_pulse(
        flip_angle_rad=np.pi / 2,
        pulse_type='excitation',
        system=pp.Opts()
    )
    rf_new_ns = rf_new.to_simple_ns()
    log_module.info("compare rf")

    grad_ns = pp.make_extended_trapezoid(
        'z',
        amplitudes=np.array([0, 755857.8987, 755857.8987, 0]),
        times=np.array([0, 0.00011, 0.00189, 0.002])
    )
    grad_new = GRAD.make_trapezoid('z', system=pp.Opts(), area=1 / 0.7 * 1e3, duration_s=2e-3)
    grad_new_ns = grad_new.to_simple_ns()
    log_module.info("compare grad")

    adc_ns = pp.make_adc(num_samples=304, duration=3e-3)
    adc_new = ADC.make_adc(system=pp.Opts(), num_samples=304, duration_s=3e-3)
    adc_new_ns = adc_new.to_simple_ns()

    log_module.info("compare adc")

    delay_ns = pp.make_delay(1e-3)
    delay_new = DELAY.make_delay(1e-3)

    log_module.info("compare delays")
