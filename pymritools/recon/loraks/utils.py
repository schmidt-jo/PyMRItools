import logging
from typing import Tuple

import torch
from torch.nn.functional import pad
from torch_kmeans import KMeans
import numpy as np
import tqdm

log_module = logging.getLogger(__name__)


def validate_input_shape(k_space_rpsct: torch.Tensor):
    """
    Validate and cast the input tensor to ensure it has 5 dimensions.

    Checks if the input tensor has 5 dimensions. If not, it will either:
    - Raise a ValueError if the tensor has more than 5 dimensions
    - Unsqueeze (add) dimensions if the tensor has less than 5 dimensions

    Args:
        k_space_rpsct (torch.Tensor): Input tensor to validate

    Returns:
        torch.Tensor: Validated tensor with 5 dimensions
            Dimensions represent: [n read, n phase, n slice, n channels, n contrasts]

    Raises:
        ValueError: If input tensor has more than 5 dimensions
    """
    # Validate input shape
    if k_space_rpsct.dim() > 5:
        raise ValueError("Input tensor must have 5 dimensions: [n read, n phase, n slice, n channels, n contrasts].")
    while k_space_rpsct.dim() < 5:
        warning = (f"Input tensor has less than 5 dimensions ({k_space_rpsct.shape})."
                   f" We unfold last dimension, might create unwanted behaviour.")
        log_module.warning(warning)
        k_space_rpsct = k_space_rpsct.unsqueeze(-1)
    log_module.info(f"Input tensor shape: {k_space_rpsct.shape}")
    return k_space_rpsct


def check_channel_batch_size_and_batch_channels(
        k_space_rpsct: torch.Tensor, batch_size_channels: int,
        use_correlation_clustering: bool = True):
    """
    Check and prepare channel batching for the input tensor.

    Validates the input tensor shape and determines channel batching strategy.

    Args:
        k_space_rpsct (torch.Tensor): Input k-space tensor
        batch_size_channels (int): Desired number of channels per batch

    Returns:
        torch.Tensor: Indices for channel batches
    """
    k_space_rpsct = validate_input_shape(k_space_rpsct=k_space_rpsct)

    in_shape = k_space_rpsct.shape
    n_channels = in_shape[-2]

    # check if we batch the channel dims
    batching, batch_size_channels, num_channel_batches = check_channel_batching(
        batch_size_channels=batch_size_channels, n_channels=n_channels
    )

    # batch channels
    if batching and batch_size_channels > 1:
        if use_correlation_clustering:
            batch_channel_indices = extract_channel_batch_indices(
                k_data_nrps_c=k_space_rpsct[..., 0], batch_size_channels=batch_size_channels
            )
            torch.cuda.empty_cache()
        else:
            torch.manual_seed(8)
            batch_channel_indices = torch.stack(
                torch.tensor_split(
                    torch.randperm(n_channels), num_channel_batches),
                dim=0
            )
    elif batching and batch_size_channels == 1:
        batch_channel_indices = torch.arange(n_channels)[:, None]
    else:
        batch_channel_indices = torch.arange(batch_size_channels)[None]

    return batch_channel_indices


def prepare_k_space_to_batches(k_space_rpsct: torch.Tensor, batch_channel_indices: torch.Tensor = None):
    """
    Prepare k-space tensor for batched processing by rearranging dimensions.

    Transforms the input k-space tensor to a batched format suitable for computation.

    Args:
        k_space_rpsct (torch.Tensor): Input k-space tensor
        batch_channel_indices (torch.Tensor, optional): Pre-computed channel batch indices

    Returns:
        Tuple[torch.Tensor, Tuple]:
            - Batched k-space tensor with dimensions [bc * ns, ncb * nt, np, nr]
            - Original input tensor shape
    """
    # Assuming channel dimension = -2
    input_shape = k_space_rpsct.shape
    n_channels = input_shape[-2]
    # batch channel_dimensions
    if batch_channel_indices is None:
        # we just take all channels as one batch
        batch_channel_indices = torch.arange(n_channels)[None]
    # we implement slice wise computation (batch slices), additionally, we might need to
    # compress channels or batch them additionally
    # thus in this method we want to end up with batched k-space: [b, nct, ny, nx],
    # where nct is combining echos and (potentially batched) channels.

    # starting dims [nr, np, ns, nc, nt]
    # move channels to front and reverse axis order
    k_data = k_space_rpsct.permute(3, 4, 2, 1, 0)
    # [nc, nt, ns, np, nr]
    # assume we checked the batching of channels already, we do the batching
    # we batch the channel dimensions based on the found batches
    k_data = k_data[batch_channel_indices]
    # we now have a tuple of channel batched k tensors
    # should be dims [bc, ncb, nt, nz, ny, nx]
    bc, ncb, nt, ns, n_phase, nr = k_data.shape
    # we pull the slice dim to front
    k_data = k_data.permute(0, 3, 1, 2, 4, 5)
    # dims [bc, ns, ncb, nt, np, nr]
    # we reshape echo and batched channels, and batch dimensions
    k_data = torch.reshape(k_data, (bc * ns, ncb * nt, n_phase, nr))
    return k_data, input_shape


def unprepare_batches_to_k_space(
        k_batched: torch.Tensor, batch_channel_indices: torch.Tensor, original_shape: Tuple):
    """
    Reverse the batching process and reconstruct the original k-space tensor.

    Args:
        k_batched (torch.Tensor): Batched k-space tensor
        batch_channel_indices (torch.Tensor): Channel batch indices
        original_shape (Tuple): Shape of the original input tensor

    Returns:
        torch.Tensor: Reconstructed k-space tensor with original dimensions
    """
    # assume that batch_channel_indices have dims [bc, ncb] of num of channel batches and channel batch size
    num_channel_batches, batch_size_channels = batch_channel_indices.shape
    # needs to reverse above batching
    # now dims [bc * ns, nt * ncb, np, nr]
    # need some information about channel batching (we might take this from above method as well?)
    k_batched = k_batched.reshape(
        (num_channel_batches, original_shape[2], batch_size_channels,
         original_shape[-1], original_shape[1], original_shape[0])
    )
    # now [bc, ns, nt, ncb, np, nr]
    # undo slice permutation
    k_batched = k_batched.permute(0, 2, 3, 1, 4, 5)
    # now dims [bc, ns, nt, ncb, np, nr]
    # allocate out data and use same permutation
    k_data_out = torch.zeros(original_shape, dtype=k_batched.dtype, device=k_batched.device).permute(3, 4, 2, 1, 0)
    # assign
    k_data_out[batch_channel_indices] = k_batched
    # reverse order
    k_data_out = k_data_out.permute(4, 3, 2, 0, 1)
    return k_data_out


def pad_input(k_space: torch.Tensor, sampling_dims: Tuple = (-2, -1)):
    """
    Pad the input tensor to ensure odd dimensions along specified sampling directions.

    Args:
        k_space (torch.Tensor): Input tensor to be padded
        sampling_dims (Tuple, optional): Dimensions to pad. Defaults to last two dimensions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Padded k-space tensor
            - Padding values used
    """
    # Convert negative dimensions to positive if needed
    dims_to_pad = [dim if dim >= 0 else k_space.ndim + dim for dim in sampling_dims]

    # Create padding list (torch.nn.functional.pad expects reversed [xn1, xn2, ...., x11 x12, x01, x02] order)
    padding = [0] * (2 * k_space.ndim)

    for dim in dims_to_pad:
        # Check if the dimension is even
        if k_space.shape[dim] % 2 == 0:
            # Pad 0 at the end of this dimension
            padding[2 * (k_space.ndim - dim) - 1] = 1

    # Apply padding
    k_padded = pad(k_space, pad=padding, mode='constant', value=0.0)

    return k_padded, tuple(padding)


def unpad_output(k_space: torch.Tensor, padding: Tuple[int, int]):
    """
    Remove padding from the output tensor.

    Args:
        k_space (torch.Tensor): Padded tensor
        padding (Tuple[int, int]): Padding values to remove

    Returns:
        torch.Tensor: Unpadded tensor
    """
    # padding has reversed order
    padding = torch.tensor(padding).__reversed__()
    for i in range(len(k_space.shape)):
        if not (padding[2*i:2*(i+1)] == 0).all():
            start = padding[2*i+1]
            end = k_space.shape[i] - padding[2*i]
            k_space = torch.movedim(k_space, i, 0)
            k_space = k_space[start:end]
            k_space = torch.movedim(k_space, 0, i)
    return k_space


def check_channel_batching(batch_size_channels: int, n_channels: int) -> (bool, torch.Tensor, torch.Tensor):
    """
    Validate and determine channel batching strategy.

    Args:
        batch_size_channels (int): Desired number of channels per batch
        n_channels (int): Total number of channels

    Returns:
        Tuple[bool, int, int]:
            - Whether batching is necessary
            - Actual batch size
            - Number of batches

    Raises:
        AttributeError: If channel dimensions are not compatible with batching
    """
    # check batching
    if 1 <= batch_size_channels < n_channels:
        if n_channels % batch_size_channels > 1e-9:
            msg = (
                f"Channel dimension must be divisible by channel batch-size, "
                f"otherwise batched matrices will have varying dimensions and "
                f"possibly would need varying rank settings."
            )
            log_module.error(msg)
            raise AttributeError(msg)
        num_batches: int = n_channels // batch_size_channels
        log_module.info(f"Using batched channel dimension (size: {batch_size_channels} / {n_channels})")
        batch = True
    else:
        num_batches = 1
        batch_size_channels: int = n_channels
        batch = False
    return batch, batch_size_channels, num_batches


def select_most_correlated_channels(
        correlation_matrix: torch.Tensor, channel_batch: torch.Tensor, batch_size_channels: int) -> torch.Tensor:
    """
    Select the most correlated channels within a given batch.

    Args:
        correlation_matrix (torch.Tensor): Matrix of channel correlations
        channel_batch (torch.Tensor): Current channel batch
        batch_size_channels (int): Desired number of channels per batch

    Returns:
        torch.Tensor: Indices of most correlated channels
    """
    # If too many, select the batch_size most correlated channels within this cluster
    # Find all channels in the current batch
    sub_corr = correlation_matrix[channel_batch][:, channel_batch]

    # Compute mean correlation of channels in current batch
    batch_mean_correlation = torch.mean(torch.abs(sub_corr), dim=1)

    # Select most correlated channels to pad the batch
    _, top_indices = torch.topk(batch_mean_correlation, batch_size_channels)

    return channel_batch[top_indices]


def pad_with_most_correlated_channels(
        correlation_matrix: torch.Tensor, channel_batch: torch.Tensor, batch_size_channels: int,
        available_channels_idx: torch.Tensor) -> torch.Tensor:
    """
    Pad the channel batch with additional most correlated channels.

    Args:
        correlation_matrix (torch.Tensor): Matrix of channel correlations
        channel_batch (torch.Tensor): Current channel batch
        batch_size_channels (int): Desired number of channels per batch
        available_channels_idx (torch.Tensor): Indices of available channels

    Returns:
        torch.Tensor: Padded channel batch with most correlated channels
    """

    # build a mask of all remaining channels to choose from,
    # not take the ones already identified to belong to the current batch
    remaining_channels = torch.tensor([ac for ac in available_channels_idx if ac not in channel_batch]).to(
        dtype=torch.int, device=correlation_matrix.device)

    # compute mean correlation of remaining channels to current batch
    batch_mean_correlation = torch.mean(
        torch.abs(correlation_matrix[remaining_channels][:, channel_batch]),
        dim=1
    )

    # select most correlated channels to pad the batch
    _, additional_indices = torch.topk(batch_mean_correlation, batch_size_channels - len(channel_batch))

    # combine current batch with additional channels
    return torch.cat(
        [channel_batch, remaining_channels[additional_indices]]
    )


def extract_channel_batch_indices(k_data_nrps_c: torch.Tensor, batch_size_channels: int) -> torch.Tensor:
    """
    Extract channel batch indices based on correlation clustering.

    Performs k-means clustering on channel correlations to group similar channels.

    Args:
        k_data_nrps_c (torch.Tensor): Input k-space data tensor
        batch_size_channels (int): Desired number of channels per batch

    Returns:
        torch.Tensor: Tensor of channel batch indices
    """
    nc = k_data_nrps_c.shape[-1]
    # ensure flatten for ech channel
    k_data_nrps_c = torch.reshape(k_data_nrps_c, (-1, nc))
    # compute correlation matrix
    channel_corr = torch.abs(torch.corrcoef(k_data_nrps_c.mT))

    # convert to distance matrix, higher correlation is shorter distance
    distance_matrix = 1 - channel_corr

    # set batch size
    num_batches = int(np.ceil(nc // batch_size_channels))

    # enumerate all channels, build clusters based on available channels
    available_channels = torch.arange(nc, device=k_data_nrps_c.device)
    batches = []

    for idx_c in tqdm.trange(num_batches - 1, desc="Process correlation based channel batching"):
        # each iteration we cluster the remaining elements
        kmeans = KMeans(n_clusters=num_batches - idx_c, verbose=False)
        # use kmeans clustering to group similar elements
        # perform clustering
        # get cluster labels for each channel, torch_kmeans assumes batch dim, so squeeze and unsqueeze dim 0
        labels = torch.squeeze(kmeans.fit_predict(distance_matrix[available_channels][:, available_channels][None]))
        # Create batches based on cluster assignments

        # Find channels belonging to the first cluster
        batch_channels = torch.where(labels == labels[0])[0]
        # find original indices of those channels
        batch_channels = available_channels[batch_channels]

        # Ensure we have exactly batch_size channels
        if len(batch_channels) > batch_size_channels:
            batch_channels = select_most_correlated_channels(
                correlation_matrix=channel_corr, channel_batch=batch_channels,
                batch_size_channels=batch_size_channels
            )
        elif len(batch_channels) < batch_size_channels:
            # If too few, pad with most similar additional channels
            batch_channels = pad_with_most_correlated_channels(
                correlation_matrix=channel_corr, channel_batch=batch_channels,
                available_channels_idx=available_channels, batch_size_channels=batch_size_channels
            )

        batches.append(batch_channels.tolist())
        # remove from available channels
        available_channels = torch.Tensor([int(ac) for ac in available_channels if ac not in batch_channels]).to(
            dtype=torch.int, device=k_data_nrps_c.device)

    # remaining list is the last batch
    batches.append(available_channels.tolist())
    # return as tensor
    return torch.Tensor(batches).to(torch.int)