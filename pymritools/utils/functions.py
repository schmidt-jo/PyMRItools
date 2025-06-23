import numpy as np
import torch


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Interpolates a function fp at points xp in a multidimensional context

    Parameters:
    x (torch.Tensor): Tensor of the new sampling points with shape [batch, a, b]
    xp (torch.Tensor): 1D Tensor of original sample points with shape [c]
    fp (torch.Tensor): 2D Tensor of function values at xp with shape [a, c]

    Returns:
    torch.Tensor: Interpolated values with shape [batch, a, b]
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
    Normalizes input tensor along a specified dimension using its L2-norm and handles cases
    with NaN or infinity values.

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
    Performs an N-dimensional Fourier transform on the input tensor or array, with appropriate 
    shifting to move the zero-frequency component to the center. This is useful for converting 
    frequency domain data to an image representation or for processing data that has been stored 
    in the Fourier domain.

    Args:
        input_data: The input data to be transformed.
        dims: The dimensions over which the FFT operation should be applied.
            Defaults to (-1, -2).

    Returns:
        np.ndarray | torch.Tensor: The transformed data with zero-frequency components shifted 
        to the center. The output format matches the type of the input (`np.ndarray` if input 
        is a NumPy array, `torch.Tensor` if input is a PyTorch tensor).
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
        dim_channel (int, optional): The dimension along which to compute the RSS. Defaults to -1.

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

    The function generates a Gaussian window of a given size with a specified
    standard deviation (sigma). Optionally, the user can provide the center
    of the Gaussian; if not provided, the center is calculated as the middle
    of the window.

    Args:
        size: The size of the Gaussian window.
        sigma: The standard deviation of the Gaussian.
        center: The center point of the Gaussian. If not specified, the center is set to the middle of the window based
            on the size.

    Returns:
        torch.Tensor: A 1-dimensional tensor representing the Gaussian window.

    Raises:
        None
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
