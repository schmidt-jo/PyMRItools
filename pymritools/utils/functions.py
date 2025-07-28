import numpy as np
import torch


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Interpolates a function fp at points xp in a multidimensional context

    Parameters:
    x: Tensor of the new sampling points with shape [batch, a, b]
    xp: 1D Tensor of original sample points with shape [c]
    fp: 2D Tensor of function values at xp with shape [a, c]

    Returns:
        Interpolated values with shape [batch, a, b]
    """
    while len(x.shape) < 3:
        # in case there is no batch dim we fill
        x = x.unsqueeze(0)
    batch, a, b = x.shape
    # find the closest upper adjacent indices of x in xp, then the next lower one
    indices = torch.searchsorted(xp, x.view(-1, b))
    indices = torch.clamp(indices, 1, xp.shape[0] - 1)
    # find adjacent left and right points on originally sampled axes xp
    x0 = xp[indices - 1]
    x1 = xp[indices]
    # find values of the originally sampled function considering its differing for each idx_a
    fp_expanded = fp.unsqueeze(0).expand(batch, -1, -1)
    y0 = fp_expanded.gather(2, indices.view(batch, a, b) - 1)
    y1 = fp_expanded.gather(2, indices.view(batch, a, b))
    # get the slope
    slope = (y1 - y0) / (x1 - x0).view(batch, a, b)
    interpolated_values = slope * (x - x0.view(batch, a, b)) + y0
    return interpolated_values


def normalize_data(data: torch.Tensor, dim_t: int = -1) -> (torch.Tensor, torch.Tensor):
    """
    Normalizes input tensor along a specified dimension using its L2-norm and handles cases with NaN or infinity values.

    Args:
        data: Input tensor to be normalized.
        dim_t: Dimension along which to calculate the norm. Defaults to -1 (the last dimension).

    Returns:
        A tuple containing:
            - The normalized tensor, where the L2-norm divides each element along the specified dimension.
            - The L2-norm tensor along the specified dimension.
    """
    norm_factor = torch.linalg.norm(data, dim=dim_t, keepdim=True)
    norm_data = torch.nan_to_num(
        torch.divide(data, norm_factor),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    return norm_data, torch.squeeze(norm_factor)


def fft_to_img(
        input_data: np.ndarray | torch.Tensor,
        dims: tuple | int = (-1, -2)) -> (np.ndarray | torch.Tensor):
    """
    Performs an N-dimensional Fourier transform on the input tensor or array, with appropriate shifting to move the
    zero-frequency component to the center. This is useful for converting frequency domain data to an image
    representation or for processing data that has been stored in the Fourier domain.

    Args:
        input_data: The input data to be transformed.
        dims: The dimensions over which the FFT operation should be applied. Defaults to (-1, -2).

    Returns:
        The transformed data with zero-frequency components shifted to the center.
        The output matches the type of the input.
    """
    if isinstance(dims, int):
        dims = (dims,)
    if torch.is_tensor(input_data):
        return torch.fft.fftshift(
            torch.fft.fftn(
                torch.fft.ifftshift(
                    input_data,
                    dim=dims
                ),
                dim=dims
            ),
            dim=dims
        )
    else:
        return np.fft.fftshift(
            np.fft.fftn(
                np.fft.ifftshift(
                    input_data,
                    axes=dims
                ),
                axes=dims
            ),
            axes=dims
        )


def ifft_to_k(
        input_data: np.ndarray | torch.Tensor,
        dims: tuple | int = (-1, -2)) -> (np.ndarray | torch.Tensor):
    """
    Performs an inverse FFT on the input data and shifts the zero-frequency component
    to the center of the spectrum for k-space conversion.

    Args:
        input_data: The input array or tensor containing data for which the inverse FFT needs to be computed.
        dims: A tuple or integer specifying the axes along which the inverse FFT is computed. Defaults to (-1, -2).

    Returns:
        The transformed input data after applying the inverse FFT and necessary shifts. The result has the same type
        as the input_data.
    """
    if isinstance(dims, int):
        dims = (dims,)
    if torch.is_tensor(input_data):
        return torch.fft.fftshift(
            torch.fft.ifftn(
                torch.fft.ifftshift(
                    input_data,
                    dim=dims
                ),
                dim=dims
            ),
            dim=dims
        )
    else:
        return np.fft.fftshift(
            np.fft.ifftn(
                np.fft.ifftshift(
                    input_data,
                    axes=dims
                ),
                axes=dims
            ),
            axes=dims
        )


def root_sum_of_squares(input_data: np.ndarray | torch.Tensor, dim_channel: int = -1) -> (np.ndarray | torch.Tensor):
    """
    Computes the root sum of squares (RSS) along a specified channel dimension for the input data.

    Args:
        input_data: The input data to compute the RSS for.
        dim_channel: The dimension along which to compute the RSS. Defaults to -1.

    Returns:
        The computed root sum of squares. The output type is the same as the input type.
    """
    if torch.is_tensor(input_data):
        return torch.sqrt(
            torch.sum(
                torch.abs(input_data) ** 2,
                dim=dim_channel
            )
        )
    else:
        return np.sqrt(
            np.sum(
                np.abs(input_data) ** 2,
                axis=dim_channel
            )
        )


def gaussian_window(size: int, sigma: float, center: int = None) -> torch.Tensor:
    """
    Creates a 1-dimensional Gaussian window.

    The function generates a Gaussian window of a given size with a specified standard deviation (sigma).
    Optionally, the user can provide the center of the Gaussian; if not provided, the center is calculated as the middle
    of the window.

    Args:
        size: The size of the Gaussian window.
        sigma: The standard deviation of the Gaussian.
        center: The center point of the Gaussian. If not specified, the center is set to the middle of the window based
            on the size.

    Returns:
        A 1-dimensional tensor representing the Gaussian window.
    """
    if center is None:
        center = int(np.round(size / 2))
    ax = torch.arange(size).float()
    return torch.exp(-((ax - center) ** 2) / (2 * sigma ** 2))


def gaussian_2d_kernel(size_x: int, size_y: int, center_x: int = None, center_y: int = None,
                       sigma: float | tuple[float, float] = 1.0):
    """
    Generates a 2D Gaussian kernel.

    Args:
        size_x: Kernel size in the x dimension.
        size_y: Kernel size in the y dimension.
        center_x: Center of the Gaussian in the x dimension. If None, defaults to the center of the kernel.
        center_y: Center of the Gaussian in the y dimension. If None, defaults to the center of the kernel.
        sigma: Standard deviation of the Gaussian. If a tuple is provided, the function uses different Gaussian
            scale per dim respectively.

    Returns:
        2D Gaussian kernel
    """
    if isinstance(sigma, float | int):
        sigma = (sigma, sigma)
    gauss_x = gaussian_window(size_x, sigma[0], center=center_x)
    gauss_y = gaussian_window(size_y, sigma[1], center=center_y)

    gauss_2d = torch.matmul(
        gauss_x.unsqueeze(-1), gauss_y.unsqueeze(0)
    )
    gauss_2d /= gauss_2d.sum()
    return gauss_2d


class SimpleKalmanFilter:
    """A simple implementation of a Kalman filter for estimating a value.

    The Kalman filter is used to provide a more accurate estimate of a value by combining predictions from a model and
    noisy observations, commonly used for filtering noise in measurements or tracking signals.
    """

    def __init__(self, process_variance=1e-5, measurement_variance=1e-3):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = None
        self.error_estimate = 1

    def update(self, measurement):
        """
        Updates the estimate of the system state based on the new measurement using the Kalman Filter algorithm.
        It calculates the Kalman Gain, updates the current estimate based on the measurement,
        and adjusts the error estimate accordingly.

        Args:
            measurement: The new measurement data to incorporate into the current
                state estimate.

        Returns:
            The updated state estimate after incorporating the new measurement.
        """
        if self.estimate is None:
            self.estimate = measurement
            return self.estimate

        # Kalman gain calculation
        kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_variance)

        # Estimate update
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)

        # Error estimate update
        self.error_estimate = (1 - kalman_gain) * self.error_estimate + abs(
            measurement - self.estimate) * self.process_variance

        return self.estimate


def exponential_moving_average(new_value: float, previous_smoothed: float, alpha: float = 0.1) -> float:
    """
    Calculates the exponential moving average of a time series data point.

    This function computes the exponentially weighted moving average given a new data point, the previous smoothed
    value, and a smoothing factor (alpha). It is often used in signal processing and time series analysis to produce
    a smoothed dataset.

    Args:
        new_value: Current data point in the time series.
        previous_smoothed: Previously calculated smoothed value.
        alpha: Smoothing factor, a value between 0 and 1, which determines the weight given to new data relative to
            the previous smoothed value. Defaults to 0.1.

    Returns:
        The updated smoothed value calculated as the exponential moving average.
    """
    return alpha * new_value + (1 - alpha) * previous_smoothed


def unwrap_phase(phase_data: torch.Tensor) -> torch.Tensor:
    log_module.info("Unwrapping phase image.")
    if phase_data.max() > 3.15:
        if phase_data.min() >= 0:
            norm_phase = ((phase_data / phase_data.max()) * 2 * np.pi) - np.pi
        else:
            norm_phase = (phase_data / phase_data.max()) * np.pi
    else:
        norm_phase = phase_data

    dim = norm_phase.shape

    tmp = torch.tensor(
        np.array(range(int(np.floor(-dim[1] / 2)), int(np.floor(dim[1] / 2)))) / float(dim[1])
    )
    tmp = tmp.reshape((1, dim[1]))
    uu = np.ones((1, dim[0]))
    xx = np.dot(tmp.conj().T, uu).conj().T
    tmp = np.array(
        np.array(range(int(np.floor(-dim[0] / 2)), int(np.floor(dim[0] / 2)))) / float(dim[0])
    )
    tmp = tmp.reshape((1, dim[0]))
    uu = np.ones((dim[1], 1))
    yy = np.dot(uu, tmp).conj().T
    kk2 = xx**2 + yy**2
    hp1 = gauss_filter(dim[0], GAUSS_STDEV, dim[1], GAUSS_STDEV)

    filter_phase = np.zeros_like(norm_phase)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(dim[2]):
            z_slice = norm_phase[:, :, i]
            lap_sin = -4.0 * (np.pi**2) * icfft(kk2 * cfft(np.sin(z_slice)))
            lap_cos = -4.0 * (np.pi**2) * icfft(kk2 * cfft(np.cos(z_slice)))
            lap_theta = np.cos(z_slice) * lap_sin - np.sin(z_slice) * lap_cos
            tmp = np.array(-cfft(lap_theta) / (4.0 * (np.pi**2) * kk2))
            tmp[np.isnan(tmp)] = 1.0
            tmp[np.isinf(tmp)] = 1.0
            kx2 = tmp * (1 - hp1)
            filter_phase[:, :, i] = np.real(icfft(kx2))

    filter_phase[filter_phase > np.pi] = np.pi
    filter_phase[filter_phase < -np.pi] = -np.pi
    filter_phase *= -1.0

    filter_obj = nib.Nifti1Image(filter_phase, phase_obj.affine, phase_obj.header)
    filter_obj.set_data_dtype(np.float32)
    return filter_obj


class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-3):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = None
        self.error_estimate = 1

    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return self.estimate

        # Kalman gain calculation
        kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_variance)

        # Estimate update
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)

        # Error estimate update
        self.error_estimate = (1 - kalman_gain) * self.error_estimate + abs(
            measurement - self.estimate) * self.process_variance

        return self.estimate


def exponential_moving_average(new_value, previous_smoothed, alpha=0.1):
    """
    Exponential Moving Average smoothing
    - alpha controls the smoothing (0 < alpha < 1)
    - Lower alpha = more smoothing, higher alpha = less smoothing
    """
    return alpha * new_value + (1 - alpha) * previous_smoothed


def calc_psnr(original_input: torch.Tensor, compressed_input: torch.Tensor) -> float:
    mse = torch.mean((original_input - compressed_input) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = torch.max(torch.abs(original_input))
    val = 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()
    return val


def calc_nmse(original_input: torch.Tensor, compressed_input: torch.Tensor) -> float:
    mse = torch.mean((original_input - compressed_input) ** 2)
    n = torch.clip(torch.var(original_input), min=1e-11)
    return mse / n
#
#
# def calc_ssim(original_input: torch.Tensor, compressed_input: torch.Tensor) -> float:
#

