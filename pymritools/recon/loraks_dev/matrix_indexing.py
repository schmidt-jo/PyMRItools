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


def get_all_idx_nd_square_patches_in_nd_shape(size: int, direction: tuple[int, ...], k_space_shape: tuple[int, ...]):
    """
        Get all linear indices for ND square patches within a given k-space, iterating along specified directional dimensions.

        This function generates all possible patches in the multidimensional tensor by shifting the patch along the chosen
        directional dimensions.

        Args:
            size: Size of the square patch.
            direction: Tuple indicating the direction in k-space, e.g., (0, 0, 1, 0, 1).
            k_space_shape: Shape of the k-space, e.g., (echos, channels, nz, ny, nx).

        Returns:
            torch.Tensor: Linear indices for all square patches.
        """
    # Step 1: Get indices for a single patch
    single_patch_lin_idxs = get_idx_nd_square_patch_in_nd_shape(
        size=size, direction=direction, k_space_shape=k_space_shape
    )

    # Step 2: Determine the shape along directional dimensions (where direction > 0)
    direction_mask = torch.tensor(direction).to(dtype=torch.bool)
    shape_direction_dims = torch.tensor(k_space_shape)[direction_mask]

    # The number of steps along each directional dimension for shifting patches
    num_steps = shape_direction_dims - size + 1  # Subtract patch size (valid positions for top-left corner)

    # Step 3: Compute offsets for all patches along each direction
    dim_multipliers = torch.cumprod(torch.tensor((*k_space_shape[1:], 1)).flip(0), dim=0).flip(0)
    direction_steps = dim_multipliers[direction_mask]

    # Generate all grid offsets for valid shifts in directional dimensions
    grid_coords = torch.meshgrid(*[torch.arange(n) for n in num_steps], indexing="ij")
    grid_offsets = sum((coords * step for coords, step in zip(grid_coords, direction_steps)))

    # Step 4: Combine single patch indices with grid offsets
    all_patches_indices = single_patch_lin_idxs.view(1, -1) + grid_offsets.view(-1, 1)

    # Return indices for all patches
    return all_patches_indices


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
        direction=direction,
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