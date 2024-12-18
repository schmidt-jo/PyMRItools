import torch

from pymritools.recon.loraks_dev.matrix_indexing import reshape_to_patches, restore_original_shape

def test_show_patch_shapes():
    tensor = torch.randn(2, 4, 8, 16, 16, 16)  # Beispiel-Tensor
    size = 4
    direction = (0, 0, 1, 1, 0, 0)
    k_space_shape = tensor.shape[3:]
    l_per_partition = 32

    patched_tensor = reshape_to_patches(tensor, size, direction, k_space_shape, l_per_partition)
    print("Patched Tensor Shape:", patched_tensor.shape)

    restored_tensor = restore_original_shape(patched_tensor, tensor.shape, patched_tensor.shape[0])
    print("Restored Tensor Shape:", restored_tensor.shape)