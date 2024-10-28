import logging
from abc import ABC, abstractmethod

import torch

log_module = logging.getLogger(__name__)


def get_idx_2d_grid_circle_within_radius(radius: int, device: torch.device = torch.get_default_device()) -> torch.Tensor:
    """
    Generates indices of all grid points within a radius on a 2d grid.

    NOTE: This function cannot be compiled without taking care.
    The reason is that the created circular mask will be used to create the result tensor which has a non-static size.
    The size of the tensor depends on how many points are within the radius, and this cannot be known upfront.
    Therefore, the torch compiler will complain, and if we really need to compile this function, we need to set:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    I'm sure the setting can also be provided using the @torch.compile annotation but need to check.
    -> If used with LORAKS or PCA denoising the matrix sizes explode quite quickly.
        Thus we could compile this function for a range of radii.
        Radius 1 - 2 are probably not needed, radius > 5 is probably not feasible,
        i.e. resulting in three function compiles (for R= 3-5)
    :param radius: Radius of the 2d circle.
    :param device: Desired device of the returned tensor.
    :return: Tensor with shape (#pts, 2) of x-y-points within radius
    """
    axis = torch.arange(-radius, radius + 1, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    mask = (xx ** 2 + yy ** 2 <= radius ** 2)
    # TODO: Decide what method to use. Both should be equivalent
    # return torch.stack([xx[mask], yy[mask]], dim=1)
    return torch.nonzero(mask) - radius


def get_idx_2d_rectangular_grid(size_x: int, size_y: int, device: torch.device = torch.get_default_device()) -> torch.Tensor:
    """
    Generate indices of grid points to form a rectangular 2d grid with sides size_x, size_y.
    :param size_x: Side length of rectangular grid in first dimension.
    :param size_y: Side length of rectangular grid in second dimension.
    :param device: Desired device of the returned tensor.
    :return: Tensor with shape (#pts, 2) of x-y-points of the grid.
    """
    x = torch.arange(0, size_x, device=device)
    y = torch.arange(0, size_y, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    return grid


def get_idx_2d_square_neighborhood_patches_in_shape(
        shape_2d: tuple[int, int],
        nb_size: int,
        device: torch.device = torch.get_default_device()) -> torch.Tensor:
    """
    Generates indices of all square patches of neighboring voxels in a neighborhood of size nb_size,
    on a 2d grid.
    :param shape_2d: Shape of 2d grid.
    :param nb_size: Neighborhood size of the square patches.
    :param device: Desired device of the returned tensor.
    :return: Tensor with shape (#pts, 2) of x-y-points of the grid.
    """
    # build indices of grid for whole shape
    shape_grid = get_idx_2d_rectangular_grid(
        size_x=shape_2d[0] - nb_size, size_y=shape_2d[1] - nb_size, device=device
    )
    # get indices of grid for the neighborhood
    nb_grid = get_idx_2d_rectangular_grid(size_x=nb_size, size_y=nb_size, device=device)
    # build index grid
    grid = shape_grid[:, None, :] + nb_grid[None, :, :]
    return grid



def get_idx_2d_circular_neighborhood_patches_in_shape(
        shape_2d: tuple[int, int],
        nb_radius: int,
        device: torch.device = torch.get_default_device()) -> torch.Tensor:
    """
    Generates indices of all circular patches within a 2d shape (shape_2d) of neighboring voxels
    in a neighborhood of radius nb_radius, on a 2d grid.
    :param shape_2d: Shape of 2d grid.
    :param nb_radius: Radius of the circular neighborhood.
    :param device: Desired device of the returned tensor.
    :return: Tensor with shape (#pts, 2) of x-y-points of the grid.
    """
    # build indices of grid for whole shape
    shape_grid = get_idx_2d_rectangular_grid(
        size_x=shape_2d[0] - 2 * nb_radius, size_y=shape_2d[1] - 2 * nb_radius, device=device
    ) + nb_radius
    # get indices of neighborhood grid
    nb_grid = get_idx_2d_grid_circle_within_radius(radius=nb_radius, device=device)
    # build index grid
    grid = shape_grid[:, None, :] + nb_grid[None, :, :]
    return grid


def get_idx_2d_rectangular_neighborhood_patches_in_shape(
        shape_2d: tuple[int, int],
        nb_size_x: int,
        nb_size_y: int,
        device=torch.get_default_device()) -> torch.Tensor:
    """
    Generates indices of all rectangular patches within a 2d shape (shape_2d) of neighboring voxels
    in a rectangular neighborhood with side length x and y, on a 2d grid.
    :param shape_2d: Shape of 2d grid.
    :param nb_size_x: Size of the neighborhood in x direction.
    :param nb_size_y: Size of the neighborhood in y direction.
    :param device: Desired device of the returned tensor.
    :return: Tensor with shape (#pts, 2) of x-y-points of the grid.
    """
    # build indices of grid for whole shape
    shape_grid = get_idx_2d_rectangular_grid(
        size_x=shape_2d[0] - nb_size_x + 1, size_y=shape_2d[1] - nb_size_y + 1, device=device
    )
    # get neighborhood grid
    nb_grid = get_idx_2d_rectangular_grid(nb_size_x, nb_size_y, device=device)
    # build index grid
    grid = shape_grid[:, None, :] + nb_grid[None, :, :]
    return grid

# TODO: Remove? We don't want some complex class structure initially.
class MatrixOperatorLowRank2D(ABC):
    """
    Base implementation of matrix operator,
    We want to implement an operator for data [x, y, ch, t], that operates on 2d shapes.
    ToDo: check batching of z-dims and most logic implementation for usecases (LORAKS and PCA Denoise)
    Merging of channel and time data allows for complementary sampling schemes per echo and
    is supposed to improve performance.
    This is the common base class
    """
    def __init__(self, k_space_dims_x_y_ch_t: tuple, nb_radius: int = 3,
                 device: torch.device = torch.get_default_device()):
        # save params
        self.radius: int = nb_radius
        self.k_space_dims: tuple = self._expand_dims_to_x_y_ch_t(k_space_dims_x_y_ch_t)
        self.device: torch.device = device

        # calculate the shape for combined x-y and ch-t dims
        self.reduced_k_space_dims = (
            k_space_dims_x_y_ch_t[0] * k_space_dims_x_y_ch_t[1],  # xy
            k_space_dims_x_y_ch_t[2] * k_space_dims_x_y_ch_t[3]  # ch - t
        )
        # need to build psp once with ones, such that we can extract it from the method
        self.p_star_p: torch.Tensor = torch.ones(
            (*self.k_space_dims[:2], self.reduced_k_space_dims[-1]), dtype=torch.int
        )
        self.neighborhood_indices: torch.Tensor = self._get_neighborhood_indices()
        self.neighborhood_indices_pt_sym: torch.Tensor = self._get_neighborhood_indices_point_sym()
        # update p_star_p, want this to be 2d + reduced last dim
        self.p_star_p = torch.reshape(
            torch.abs(self._get_p_star_p()), (*self.k_space_dims[:2], self.reduced_k_space_dims[-1])
        )

    @staticmethod
    def _expand_dims_to_x_y_ch_t(in_data: torch.Tensor | tuple):
        if torch.is_tensor(in_data):
            shape = in_data.shape
            while shape.__len__() < 4:
                # want the dimensions to be [x, y, ch, t], assume to be lacking time and or channel information
                in_data = in_data.unsqueeze(-1)
                shape = in_data.shape
        else:
            while in_data.__len__() < 4:
                # want the dimensions to be [x, y, ch, t], assume to be lacking time and or channel information
                in_data = (*in_data, 1)
            shape = in_data
        if shape.__len__() > 4:
            err = f"Operator only implemented for <= 4D data."
            log_module.error(err)
            raise AttributeError(err)
        return in_data

    @abstractmethod
    def _get_neighborhood_indices(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_neighborhood_indices_point_sym(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def neighborhood_size(self):
        return NotImplementedError

    def operator(self, k_space_x_y_ch_t: torch.Tensor) -> torch.Tensor:
        """ k-space input in 4d, [x, y, ch, t]"""
        # check for correct data shape, expand if necessary
        k_space_x_y_ch_t = self._expand_dims_to_x_y_ch_t(k_space_x_y_ch_t)
        return self._operator(k_space_x_y_ch_t)

    def operator_adjoint(self, x_matrix: torch.tensor) -> torch.tensor:
        return torch.squeeze(self._adjoint(x_matrix=x_matrix))

    @abstractmethod
    def _operator(self, k_space: torch.tensor) -> torch.tensor:
        """ to be implemented for each loraks type mode"""
        raise NotImplementedError

    @abstractmethod
    def _adjoint(self, x_matrix: torch.tensor) -> torch.tensor:
        raise NotImplementedError

    def _get_p_star_p(self):
        return self.operator_adjoint(
            self.operator(
                torch.ones(self.k_space_dims, dtype=torch.complex128)
            )
        )
