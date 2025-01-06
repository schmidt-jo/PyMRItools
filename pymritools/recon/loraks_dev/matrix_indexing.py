import torch

import torch


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
        raise ValueError("Input shapes must have the same length")

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


def get_idx_nd_square_patch_in_nd_shape(
        size: int, direction: tuple[int, ...], k_space_shape: tuple[int, ...]) -> torch.Tensor:
    """
    Generate linear indices for square patches based on size, direction and k-space shape.

    This is a general function for generating linear indices that allows for placing the 2d square patch along any
    two dimensions of a possibly multidimensional k-space.
    Args:
        size: The size of the square patch
        direction: Tuple indicating the direction in k-space, e.g., (0, 0, 1, 0, 1)
        k_space_shape: Shape of the k-space, e.g., (echos, channels, nz, nx, ny)

    Returns:
        torch.Tensor: Flattened indices for the square patch
    """
    dim_multipliers = torch.cumprod(torch.tensor((*k_space_shape[1:], 1)).flip(0), dim=0)
    direction_vector = torch.tensor(direction).flip(0)
    step_sizes = dim_multipliers*direction_vector
    indices = torch.arange(size).view(1, -1)
    valid_steps = step_sizes[step_sizes != 0].view(-1, 1)
    offset_matrix = indices * valid_steps
    horizontal = offset_matrix[0].view(1, -1)
    vertical = offset_matrix[1].view(-1, 1)
    return torch.flatten(horizontal + vertical)


def get_all_idx_nd_square_patches_in_nd_shape(
        size: int,
        patch_direction: tuple[int, ...],
        k_space_shape: tuple[int, ...],
        combination_direction: tuple[int, ...] = ()
):
    """
    Get all linear indices for ND square patches within a given k-space, iterating along specified directional dimensions,
    combining other dimensions along a secondary axis, and grouping unspecified dimensions along the first axis.

    Args:
        size: Size of the square patch.
        patch_direction: Tuple indicating the direction in k-space, e.g., (0, 0, 1, 0, 1).
        k_space_shape: Shape of the k-space, e.g., (echos, channels, nz, ny, nx).
        combination_direction: Tuple indicating other dimensions to combine patches across.

    Returns:
        torch.Tensor: Linear indices with specified output dimensions:
                      [unspecified, valid points, patch dimensions * combined dims].
        tuple: Reshape info for the input tensor (recommended reshaped shape).
    """
    # Step 1: Determine unspecified dimensions
    direction_mask = torch.tensor(patch_direction).to(dtype=torch.bool)
    combination_mask = torch.tensor(combination_direction).to(dtype=torch.bool)
    specified_mask = direction_mask | combination_mask  # Union of direction and combination
    unspecified_mask = ~specified_mask  # Dimensions not in direction or combination

    # Unspecified dimension shapes
    shape_unspecified_dims = torch.tensor(k_space_shape)[unspecified_mask]
    unspecified_steps = torch.cumprod(torch.tensor((*shape_unspecified_dims[1:], 1)).flip(0), dim=0).flip(0)

    # Step 2: Get indices for a single patch
    single_patch_lin_idxs = get_idx_nd_square_patch_in_nd_shape(
        size=size, direction=patch_direction, k_space_shape=k_space_shape
    )

    # Step 3: Determine valid points along directional dimensions
    shape_direction_dims = torch.tensor(k_space_shape)[direction_mask]
    num_steps = shape_direction_dims - size + 1  # Valid positions for top-left corners

    # Compute offsets for patches along directional dimensions
    dim_multipliers = torch.cumprod(torch.tensor((*k_space_shape[1:], 1)).flip(0), dim=0).flip(0)
    direction_steps = dim_multipliers[direction_mask]
    grid_coords = torch.meshgrid(*[torch.arange(n) for n in num_steps], indexing="ij")
    grid_offsets = sum((coords * step for coords, step in zip(grid_coords, direction_steps)))

    # Combine grid_offsets with single_patch_lin_idxs
    all_patches_indices = single_patch_lin_idxs.view(1, -1) + grid_offsets.view(-1, 1)

    # Step 4: Handle combination_direction
    if combination_direction:
        shape_combined_dims = torch.tensor(k_space_shape)[combination_mask]
        # Steps for combining patches along combination dimensions
        combination_steps = dim_multipliers[combination_mask]
        combination_coords = torch.meshgrid(*[torch.arange(n) for n in shape_combined_dims], indexing="ij")
        combination_offsets = sum((coords * step for coords, step in zip(combination_coords, combination_steps)))

        # Expand and fold combined dimensions into the patch dimension
        combined_patch_indices = (
                all_patches_indices.unsqueeze(0) + combination_offsets.view(-1, 1, 1)
        )
        all_patches_indices = combined_patch_indices.permute(1, 0, 2).reshape(
            combined_patch_indices.size(1), -1
        )

    # Step 5: Handle unspecified dimensions
    if unspecified_mask.any():
        # Generate combinations for unspecified dimensions
        unspecified_coords = torch.meshgrid(*[torch.arange(n) for n in shape_unspecified_dims], indexing="ij")
        unspecified_offsets = sum((coords * step for coords, step in zip(unspecified_coords, unspecified_steps)))

        # Expand along the unspecified dimension and move it to the first axis
        all_patches_indices = (
                all_patches_indices.unsqueeze(0) + unspecified_offsets.view(-1, 1, 1)
        )

    # Final reshape to match dimensions: [unspecified, valid points, patch_dims * combined_dims]
    all_patches_indices = all_patches_indices.reshape(
        shape_unspecified_dims.prod().item(), grid_offsets.numel(), -1
    )

    # Step 6: Compute reshape info for the input tensor
    reshape_shape = (
            (-1,) + tuple(s for s, mask in zip(k_space_shape, specified_mask.tolist()) if mask)
    )

    # Return indices for all patches and reshape info
    return all_patches_indices, reshape_shape
