import torch


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


def combine_indices(
        m: int, ne: int, nc: int,
        l_per_partition: int
):
    """
    Erstellt Partitionierungen der Dimensionen `m`, `ne`, `nc` in Kombinationen von `k` Partitionen.
    Args:
        m (int): Dimension `m`.
        ne (int): Dimension `ne`.
        nc (int): Dimension `nc`.
        l_per_partition (int): Größe `l` pro Partition (benutzerdefiniert).
    Returns:
        torch.Tensor: Indizes (Mapping), um Dimensionen zu Partitionieren.
        int: Anzahl `k` der Partitionen.
    """
    total_size = m * ne * nc
    assert total_size % l_per_partition == 0, "Total Größe muss ein Vielfaches von l_per_partition sein."
    k_partitions = total_size // l_per_partition

    # Erstelle Indizes zum Mappen
    indices = torch.arange(total_size).view(m, ne, nc)
    indices = indices.reshape(k_partitions, l_per_partition)

    return indices, k_partitions


def reshape_to_patches(
        tensor: torch.Tensor,
        size: int,
        direction: tuple[int, ...],
        k_space_shape: tuple[int, ...],
        l_per_partition: int
):
    """
    Kombiniert Dimensionen, extrahiert Patches und reshaped den Tensor.
    Args:
        tensor: Ursprünglicher Input-Tensor (Dimensionen: [m, ne, nc, nz, ny, nx]).
        size: Patchgröße (z. B. 2D-Patch-Größe).
        direction: Richtung für die Patch-Erstellung.
        k_space_shape: Form des Tensors.
        l_per_partition: Größe `l` innerhalb einer Partition.
    Returns:
        torch.Tensor: Transformierter Tensor in [b, np, l].
    """
    # Step 1: Indexpartitionierung (für `m`, `ne`, `nc`)
    m, ne, nc = tensor.shape[:3]
    indices, k_partitions = combine_indices(m, ne, nc, l_per_partition)

    # Batchdimension (b entspricht Partitionen)
    max_patches = k_partitions

    # Step 2: Extrahiere alle Patches
    all_patches = get_all_idx_nd_square_patches_in_nd_shape(
        size=size,
        patch_direction=direction,
        k_space_shape=k_space_shape
    )

    # Step 3: Reshape für die Ausgabe [b, np, l]
    patched_tensor = tensor.view(-1, *k_space_shape[3:])  # Kombiniere m, ne, nc zu batchähnlichem Format
    patched_tensor = patched_tensor[all_patches]
    patched_tensor = patched_tensor.view(max_patches, all_patches.shape[0], l_per_partition)

    return patched_tensor


def restore_original_shape(processed_tensor: torch.Tensor, original_shape: tuple, k_partitions: int):
    """
    Transformiert verarbeitete Tensoren zurück in ihre ursprüngliche Form.
    Args:
        processed_tensor: Tensor in [b, np, l]-Form (nach Verarbeitung).
        original_shape: Ursprüngliches Tensor-Shape (z. B. [m, ne, nc, nz, ny, nx]).
        k_partitions: Anzahl `k` der Partitionen.
    Returns:
        torch.Tensor: Rücktransformierter Tensor.
    """
    m, ne, nc, nz, ny, nx = original_shape
    total_size = m * ne * nc
    l_per_partition = total_size // k_partitions

    restored = processed_tensor.view(k_partitions, -1, l_per_partition)
    restored = restored.permute(0, 2, 1).reshape(original_shape)
    return restored


def dev():
    tensor = torch.randn(2, 4, 8, 16, 16, 16)  # Beispiel-Tensor
    size = 4
    direction = (0, 0, 1, 1, 0, 0)
    k_space_shape = tensor.shape[3:]
    l_per_partition = 32

    patched_tensor = reshape_to_patches(tensor, size, direction, k_space_shape, l_per_partition)
    print("Patched Tensor Shape:", patched_tensor.shape)

    restored_tensor = restore_original_shape(patched_tensor, tensor.shape, patched_tensor.shape[0])
    print("Restored Tensor Shape:", restored_tensor.shape)


if __name__ == '__main__':
    dev()