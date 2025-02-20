import logging
import torch

log_module = logging.getLogger(__name__)


def get_linear_indices(
        k_space_shape: tuple[int, ...],
        patch_shape: tuple[int, ...],
        sample_directions: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Computes linear indices for multidimensional k-space sampling.

    This function calculates the linear indices needed for sampling patches in a
    multidimensional k-space. It supports flexible patch sizes and selective
    sampling along specified dimensions.

    Args:
        k_space_shape: A tuple of integers defining the shape of the k-space.
            Each element represents the size along the corresponding dimension.
        patch_shape: A tuple of integers indicating the patch size along each dimension.
            Use -1 to specify that the patch should fill the entire space along
            that dimension.
        sample_directions: A tuple of integers (0 or 1) determining whether sampling
            should occur along each dimension. 1 indicates sampling should occur, while
            0 indicates no sampling along that dimension.

    Returns:
        tuple:
            - torch.Tensor: Flat tensor containing the computed linear indices
              for sampling.
            - tuple[int, ...]: Shape of the output representing (number of samples,
              patch size).

    Raises:
        ValueError: If the input tuples have different lengths or contain invalid values.
        ValueError: If patch_shape contains values less than -1.
        ValueError: If sample_directions contains values other than 0 or 1.

    Examples:
        >>> k_space = (256, 256, 32)  # 3D k-space
        >>> patch = (16, 16, -1)      # patches of 16x16 covering full depth
        >>> directions = (1, 1, 0)     # sample in patch-plane only
        >>> indices, shape = get_linear_indices(k_space, patch, directions)
    """
    # Turn tuples into tensors so that we can use `torch.where` to change elements conditionally
    k_space_shape = torch.tensor(k_space_shape).flip(0)
    patch_shape = torch.tensor(patch_shape).flip(0)
    sample_directions = torch.tensor(sample_directions).flip(0)

    if len(k_space_shape.shape) != 1:
        raise ValueError("Input shapes must be 1D tuples")

    if k_space_shape.shape != patch_shape.shape or k_space_shape.shape != sample_directions.shape:
        raise ValueError(f"Input shapes must have the same length: k_space_shape {k_space_shape.shape}, "
                         f"patch_shape {patch_shape.shape}, "
                         f"sample_directions {sample_directions.shape}")

    if not torch.all((sample_directions == 0) | (sample_directions == 1)):
        raise ValueError("Sample directions must be 0 or 1")

    # This calculates the offsets we need to jump exactly one step along a certain dimension.
    dim_offsets = torch.cumprod(torch.tensor([1, *k_space_shape[:-1]]), dim=0)

    # We start by calculating the linear indices for one single patch.
    # The patch format is mostly for convenience, and first we have to understand that
    # for a patch defined only in x-y-direction, e.g. (3, 3, 0, 0, 0)
    # the size of the patch is indeed 1 in all other directions.
    # Therefore, we replace all 0 in the patch definition with 1.
    # The -1s are replaced by the size of this particular dimension because a -1 indicates that the patch should
    # "fill" the whole dimension.
    patch_sizes = torch.where(patch_shape == 0, torch.tensor(1), patch_shape)
    patch_sizes = torch.where(patch_shape == -1, k_space_shape, patch_sizes)
    patch_steps = [torch.arange(size) for size in patch_sizes]


    # The creation of patch indices follows a dimensional stacking approach to generate linear indices
    # for a multidimensional patch. Starting with indices along the first dimension, we progressively
    # incorporate additional dimensions by combining the existing indices with offset-adjusted indices
    # of each new dimension. This process is similar to how we naturally count positions in a
    # multidimensional grid - we start with positions in one dimension and then systematically add
    # offsets to account for movements in other dimensions. The offsets ensure we "jump" to the correct
    # linear position when moving along higher dimensions, just like how in a 2D array, moving one row
    # down requires adding the width of the entire row to the current position.
    patch_indices = patch_steps[0]
    for i in range(1, len(patch_steps)):
        patch_indices = (patch_indices.unsqueeze(0) + dim_offsets[i] * patch_steps[i].unsqueeze(1)).flatten()

    # With the indices for the patch, we repeat the process in a very similar fashion to sample the patch
    # along the given sample directions.
    # The difference is that, this time, we use all indices along the sample directions and we only adjust them
    # so that the patch never gets outside the bounds.
    sample_steps = torch.where(sample_directions == 0, torch.tensor(1), k_space_shape - patch_sizes + 1)
    sample_steps = [torch.arange(size) for size in sample_steps]

    sample_indices = sample_steps[0]
    for i in range(1, len(sample_steps)):
        sample_indices = (sample_indices.unsqueeze(0) + dim_offsets[i] * sample_steps[i].unsqueeze(1)).flatten()

    indices = patch_indices.unsqueeze(0) + sample_indices.unsqueeze(1)
    return indices.flatten(), (*indices.shape,)


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

