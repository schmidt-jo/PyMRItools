import logging
import torch
from torch_kmeans import KMeans
import numpy as np
import tqdm

log_module = logging.getLogger(__name__)


def prepare_k_space_to_batches(k_space: torch.Tensor, batch_size_channels: int = -1):
    """
            Prepare the input k-space tensor into batches for separate computations, adjusting for memory requirements.

            The input tensor must have the shape [nx, ny, nz, ne, nt]. The output tensor will be reshaped into
            batches with a fixed output dimensionality of [b, -1], where b is the batch size.

            :param k_space: Input k-space tensor of shape [nx, ny, nz, ne, nt].
            :raises ValueError: If the input tensor does not match the expected shape.
            :return: Output tensor divided into batches, reshaped to [b, -1].
            """
    # Validate input shape
    if k_space.dim() > 5:
        raise ValueError("Input tensor must have 5 dimensions: [nx, ny, nz, ne, nt].")
    while k_space.dim() < 5:
        warning = (f"Input tensor has less than 5 dimensions ({k_space.shape})."
                   f" We unfold last dimension, might create unwanted behaviour.")
        log_module.warning(warning)
        k_space = k_space.unsqueeze(-1)
    log_module.info(f"Input tensor shape: {k_space.shape}")

    # Assuming channel dimension = -2
    n_channels = k_space.shape[-2]
    # if the input does not fit on RAM we want to batch the channel dimension, assuming they always share the
    # sampling pattern (opposed to eg. contrasts)
    # ToDo: include Memory size checks? Might be algorithm dependent
    # check if we batch the channel dims
    batching, batch_size_channels, num_channel_batches = check_channel_batching(
        batch_size_channels=batch_size_channels, n_channels=n_channels
    )

    # batch channels


    # combine all batches


    # pad input to have odd spatial dimensions


    # move dims

    # TODO: we can include the padding here as well
    input_shape = k_space.shape

    # decide on matrix sizes, maybe from memory computations? first create matrices from channels and or contrasts.
    # thus we have spatial dimensions and some combination of channels and contrasts -> [bce, nxyz, nce]
    # ToDo: device combination strategy based on memory computations (those might differ per algorithm used),
    #  should this be an abstract method again?



    k_combined = NotImplemented
    # we flatten everything per batch as we work with linear indices
    b, nxyz, nce = k_combined.shape
    combined_shape = (b, nxyz, nce)

    k_batched = k_combined.reshape(b, -1)


def unprepare_batches_to_k_space():
    """
            This method needs to reverse the above batching
            :param k_batches:
            :return:
            """
    # unflatten the shape dim (need some shapes here...)
    # need some information about channel batching (we might take this from above method as well?)
    k_combined = k_batches.reshape(combined_shape)
    # unfold individual dimensions - might need to process some channel / echo combinations
    k_space = k_combined.reshape(input_shape)
    # TODO: we need the unpadding here
    return k_space


def check_channel_batching(batch_size_channels: int, n_channels: int) -> (bool, torch.Tensor, torch.Tensor):
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
        self, correlation_matrix: torch.Tensor, batch_channels_idx: torch.Tensor) -> torch.Tensor:
    # If too many, select the batch_size most correlated channels within this cluster
    # Find all channels in the current batch
    sub_corr = correlation_matrix[batch_channels_idx][:, batch_channels_idx]

    # Compute mean correlation of channels in current batch
    batch_mean_correlation = torch.mean(torch.abs(sub_corr), dim=1)

    # Select most correlated channels to pad the batch
    _, top_indices = torch.topk(batch_mean_correlation, self.batch_channel_size)

    return batch_channels_idx[top_indices]


def pad_with_most_correlated_channels(
        self,
        correlation_matrix: torch.Tensor, batch_channels_idx: torch.Tensor,
        available_channels_idx: torch.Tensor) -> torch.Tensor:
    # build a mask of all remaining channels to choose from,
    # not take the ones already identified to belong to the current batch
    remaining_channels = torch.tensor([ac for ac in available_channels_idx if ac not in batch_channels_idx]).to(
        torch.int)

    # compute mean correlation of remaining channels to current batch
    batch_mean_correlation = torch.mean(
        torch.abs(correlation_matrix[remaining_channels][:, batch_channels_idx]),
        dim=1
    )

    # select most correlated channels to pad the batch
    _, additional_indices = torch.topk(batch_mean_correlation, self.batch_channel_size - len(batch_channels_idx))

    # combine current batch with additional channels
    return torch.cat(
        [batch_channels_idx, remaining_channels[additional_indices]]
    )


def extract_channel_batch_indices(
        self, k_data_nxzy_c: torch.Tensor) -> torch.Tensor:
    nc = k_data_nxzy_c.shape[-1]
    # ensure flatten for ech channel
    k_data_nxzy_c = torch.reshape(k_data_nxzy_c, (-1, nc))
    # compute correlation matrix
    channel_corr = torch.abs(torch.corrcoef(k_data_nxzy_c.mT))

    # convert to distance matrix, higher correlation is shorter distance
    distance_matrix = 1 - channel_corr

    # set batch size
    num_batches = int(np.ceil(nc // self.batch_channel_size))

    # enumerate all channels, build clusters based on available channels
    available_channels = torch.arange(nc)
    batches = []

    for idx_c in tqdm.trange(num_batches - 1):
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
        if len(batch_channels) > self.batch_channel_size:
            batch_channels = self._select_most_correlated_channels(
                correlation_matrix=channel_corr, batch_channels_idx=batch_channels,
            )
        elif len(batch_channels) < self.batch_channel_size:
            # If too few, pad with most similar additional channels
            batch_channels = self._pad_with_most_correlated_channels(
                correlation_matrix=channel_corr, batch_channels_idx=batch_channels,
                available_channels_idx=available_channels
            )

        batches.append(batch_channels.tolist())
        # remove from available channels
        available_channels = torch.Tensor([int(ac) for ac in available_channels if ac not in batch_channels]).to(
            torch.int)

    # remaining list is the last batch
    batches.append(available_channels.tolist())
    # return as tensor
    return torch.Tensor(batches).to(torch.int)