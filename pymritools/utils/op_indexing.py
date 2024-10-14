import logging

import torch

log_module = logging.getLogger(__name__)


def get_idx_2d_grid_circle_within_radius(radius: int):
    """
    fn to generate indices of all grid points within a radius on a 2d grid
    :param radius: [int]
    :return: tensor, dim [#pts, 2] of pts xy within radius
    """
    # generate the neighborhood
    axis = torch.arange(-radius, radius + 1)
    # generate grid
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    # get mask of pts within radius
    mask = (xx ** 2 + yy ** 2 <= radius ** 2)
    # get indices
    return torch.stack([xx[mask], yy[mask]], dim=1)


def get_idx_2d_square_grid(size_x: int, size_y: int):
    """
    fn to generate indices of grid points to form a rectangular 2d grid with sides size_x, size_y
    :param size_x: [int] side length of rectangular grid in first dimension
    :param size_y: [int] side length of rectangular grid in second dimension
    :return: tensor, dim [#pts, 2] of pts xy
    """
    x = torch.arange(0, size_x)
    y = torch.arange(0, size_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    return grid


def get_idx_2d_square_neighborhood_patches_in_shape(shape_2d: tuple[int], nb_size: int):
    """
    fn to generate indices of all square patches of neighboring voxels in a neighborhood of size nb_size,
    on a 2d grid.
    :param shape_2d: [tuple[int, int]] shape of 2d grid.
    :param nb_size: [int]
    :return: tensor, dim [#pts, 2] of pts xy of all patches
    """
    # build indices of grid for whole shape
    shape_grid = get_idx_2d_square_grid(size_x=shape_2d[0] - nb_size, size_y=shape_2d[1] - nb_size)
    # get indices of grid for neighborhood
    nb_grid = get_idx_2d_square_grid(size_x=nb_size, size_y=nb_size)
    # build index grid
    grid = shape_grid[:, None, :] + nb_grid[None, :, :]
    return grid



def get_idx_2d_circular_neighborhood_patches_in_shape(shape_2d: tuple[int], nb_radius: int):
    """
    fn to generate indices of all circular patches within a 2d shape (shape_2d) of neighboring voxels
    in a neighborhood of radius nb_radius, on a 2d grid.
    :param shape_2d: [tuple[int, int]] shape of 2d grid.
    :param nb_radius: [int]
    :return: tensor, dim [#pts, 2] of pts xy of all patches
    """
    half_r = int(nb_radius / 2)
    # build indices of grid for whole shape
    shape_grid = get_idx_2d_square_grid(size_x=shape_2d[0] - nb_radius, size_y=shape_2d[1] - nb_radius) + half_r
    # get indices of neighborhood grid
    nb_grid = get_idx_2d_grid_circle_within_radius(radius=nb_radius)
    # build index grid
    grid = shape_grid[:, None, :] + nb_grid[None, :, :]
    return grid


def get_idx_2d_rectangular_neighborhood_patches_in_shape(shape_2d: tuple[int], nb_size_x: int, nb_size_y: int):
    """
    fn to generate indices of all rectangular patches within a 2d shape (shape_2d) of neighboring voxels
    in a rectangular neighborhood with sideength x and y, on a 2d grid.
    :param shape_2d: [tuple[int, int]] shape of 2d grid.
    :param nb_size_x: [int]
    :param nb_size_y: [int]
    :return: tensor, dim [#pts, 2] of pts xy of all patches
    """
    # build indices of grid for whole shape
    shape_grid = get_idx_2d_square_grid(size_x=shape_2d[0] - nb_size_y, size_y=shape_2d[1] - nb_size_y)
    # get neighborhood grid
    nb_grid = get_idx_2d_rectangular_neighborhood_patches_in_shape(
        shape_2d=shape_2d, nb_size_x=nb_size_x, nb_size_y=nb_size_y
    )
    # build index grid
    grid = shape_grid[:, None, :] + nb_grid[None, :, :]
    return grid


