import logging
import pathlib as plib

from abc import ABC, abstractmethod

import torch
import plotly.graph_objects as go
import plotly.subplots as psub

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


def get_flat_idx_circular_neighborhood_patches_in_shape(
        shape_2d: tuple[int, int],
        nb_radius: int,
        device: torch.device = torch.get_default_device()) -> torch.Tensor:
    """
    Generates 1d indices of all circular patches within a flattened 2d shape (shape_2d) of neighboring voxels
    in a neighborhood of radius nb_radius, on a 2d grid.
    :param shape_2d: Shape of 2d grid.
    :param nb_radius: Radius of the circular neighborhood.
    :param device: Desired device of the returned tensor.
    :return: Tensor with shape (#pts, 2) of x-y-points of the grid.
    """
    # build indices of grid for whole shape and neighborhoods
    # dims: [nx ny, nb, 2]
    index_grid = get_idx_2d_circular_neighborhood_patches_in_shape(
        shape_2d=shape_2d, nb_radius=nb_radius, device=device
    )
    # flattened indices by easy computation
    indices_1d = index_grid[:, :, 0] * shape_2d[1] + index_grid[:, :, 1]
    return indices_1d


def get_flat_idx_square_neighborhood_patches_in_shape(
        shape_2d: tuple[int, int],
        nb_size: int,
        device: torch.device = torch.get_default_device()) -> torch.Tensor:
    """
    Generates 1d indices of all circular patches within a flattened 2d shape (shape_2d) of neighboring voxels
    in a neighborhood of radius nb_radius, on a 2d grid.
    :param shape_2d: Shape of 2d grid.
    :param nb_size: side-length of the square neighborhood.
    :param device: Desired device of the returned tensor.
    :return: Tensor with shape (#pts, 2) of x-y-points of the grid.
    """
    # build indices of grid for whole shape and neighborhoods
    # dims: [nx ny, nb, 2]
    index_grid = get_idx_2d_square_neighborhood_patches_in_shape(
        shape_2d=shape_2d, nb_size=nb_size, device=device
    )
    # flattened indices by easy computation
    indices_1d = index_grid[:, :, 0] * shape_2d[1] + index_grid[:, :, 1]
    return indices_1d


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


def dev():
    # set path
    path = plib.Path("dev_sim/mat_indexing").absolute()

    # build 2d shape
    nx, ny = (120, 100)
    nb_size = 3
    init_shape = torch.zeros((nx, ny))
    # build 2d square and circular indices
    ind_sq_2d = get_idx_2d_square_neighborhood_patches_in_shape(shape_2d=(nx, ny), nb_size=nb_size+torch.sqrt(torch.pi))
    ind_circ_2d = get_idx_2d_circular_neighborhood_patches_in_shape(shape_2d=(nx, ny), nb_radius=nb_size)

    # set first and random neighborhood to values
    mat_sq_2d = init_shape.clone()
    mat_sq_2d[ind_sq_2d[0, :, 0], ind_sq_2d[0, :, 1]] = 1
    mat_sq_2d[ind_sq_2d[1000, :, 0], ind_sq_2d[1000, :, 1]] = 2
    mat_circ_2d = init_shape.clone()
    mat_circ_2d[ind_circ_2d[0, :, 0], ind_circ_2d[0, :, 1]] = 1
    mat_circ_2d[ind_circ_2d[1000, :, 0], ind_circ_2d[1000, :, 1]] = 2

    # want to build flattened 1d indexing
    ind_sq_1d = ind_sq_2d[:, :, 0] * ny + ind_sq_2d[:, :, 1]
    ind_circ_1d = ind_circ_2d[:, :, 0] * ny + ind_circ_2d[:, :, 1]

    # Use indices to set values
    mat_sq_1d = init_shape.clone().view(-1)
    mat_sq_1d[ind_sq_1d[0]] = 1
    mat_sq_1d[ind_sq_1d[1000]] = 2
    mat_circ_1d = init_shape.clone().view(-1)
    mat_circ_1d[ind_circ_1d[0]] = 1
    mat_circ_1d[ind_circ_1d[1000]] = 2

    # reshape mat_1d for plotting
    mat_sq_1d = mat_sq_1d.view(nx, ny)
    mat_circ_1d = mat_circ_1d.view(nx, ny)

    # plot
    fig = psub.make_subplots(
        rows=2, cols=2,
        row_titles=["2D indexing", "1D indexing"],
        column_titles=["Square", "Circular"]
    )
    fig.add_trace(
        go.Heatmap(z=mat_sq_2d), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=mat_sq_1d), row=2, col=1
    )
    fig.add_trace(
        go.Heatmap(z=mat_circ_2d), row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=mat_circ_1d), row=2, col=2
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig_name = path.joinpath("2d_indexing_nbs").with_suffix(".html")
    print(f"write file: {fig_name}")
    fig.write_html(fig_name)


if __name__ == '__main__':
    dev()
