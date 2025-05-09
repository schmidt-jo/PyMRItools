import torch


def get_circular_nb_indices(nb_radius):
    """
    Calculate indices for a circular neighborhood within a given radius.

    The function generates a grid of indices corresponding to the vertical
    and horizontal distances from the center. Using these indices, it determines
    the points that lie within the specified circular radius by applying a mask
    based on the Euclidean distance.

    :param nb_radius: Radius of the circular neighborhood. The returned indices
        correspond to points that are at most this distance from the center.
    :type nb_radius: int
    :return: A tensor of indices representing points within the circular
        neighborhood. Each index is a set of relative coordinates (x, y) from the
        center.
    :rtype: torch.Tensor
    """
    # want a circular neighborhood, i.e. find all indices within a radius
    nb_x, nb_y = torch.meshgrid(
        torch.arange(-nb_radius, nb_radius + 1),
        torch.arange(-nb_radius, nb_radius + 1),
        indexing='ij'
    )
    # Create a mask for the circular neighborhood
    nb_r = nb_x ** 2 + nb_y ** 2 <= nb_radius ** 2
    # Get the indices of the circular neighborhood
    return torch.nonzero(nb_r).squeeze()


def get_circular_nb_indices_in_2d_shape(k_space_2d_shape: tuple, nb_radius: int, reversed: bool = False):
    """
    Calculates circular neighborhood indices in a 2D shape and converts them to linear indices.

    This function generates circular neighborhood indices for a given radius and translates them
    into linear indices relative to the provided 2D shape dimensions. Depending on the `reversed`
    flag, the neighborhood indices can be adjusted relative to the center or flipped accordingly.

    :param k_space_2d_shape: The 2D shape of the space, as a tuple, where the first value represents
        the number of rows and the second value represents the number of columns.
    :param nb_radius: The radius of the circular neighborhood within the 2D shape.
    :param reversed: A boolean flag indicating whether the neighborhood indices should be adjusted
        relative to the reversed center. Defaults to False.
    :return: A tensor containing the linear indices of the circular neighborhood within the 2D shape.
    :rtype: torch.Tensor
    """
    # want a circular neighborhood radius and convert to linear indices
    neighborhood_indices = get_circular_nb_indices(nb_radius=nb_radius)

    # Calculate offsets relative to the center
    offsets = neighborhood_indices - nb_radius

    y, x = torch.meshgrid(
        torch.arange(k_space_2d_shape[-2]), torch.arange(k_space_2d_shape[-1]),
        indexing="ij"
    )

    yx = torch.concatenate((y.unsqueeze(-1), x.unsqueeze(-1)), dim=-1)
    yx = torch.reshape(yx, (-1, 2))

    yxnb = yx.unsqueeze(0) + offsets.unsqueeze(1)
    idx = torch.all(
        (yxnb >= torch.tensor([1 - s % 2 for s in k_space_2d_shape[:2]])) &
        (yxnb < torch.tensor(k_space_2d_shape[:2])),
        dim=(0, 2)
    )

    if reversed:
        yxnb = torch.tensor(k_space_2d_shape[:2]).unsqueeze(0).unsqueeze(1) - yx.unsqueeze(0) + offsets.unsqueeze(1) - 1
    else:
        yxnb = yx.unsqueeze(0) + offsets.unsqueeze(1)

    yxnb = yxnb[:, idx]
    # convert to linear indices
    neighborhood_linear_indices = yxnb[..., 0] * k_space_2d_shape[1] + yxnb[..., 1]
    return neighborhood_linear_indices


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
