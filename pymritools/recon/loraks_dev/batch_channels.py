import torch


def batch_channels_by_correlation(correlation_matrix, num_batches, batch_size):
    """
    Create batches of channels ensuring all channels are used exactly once.

    Parameters:
    - correlation_matrix: Torch tensor of shape [nc, nc]
    - num_batches: Number of batches to create
    - batch_size: Number of channels per batch

    Returns:
    - List of batches, where each batch is a list of channel indices
    """
    # Ensure correlation matrix is a torch tensor
    if not isinstance(correlation_matrix, torch.Tensor):
        correlation_matrix = torch.tensor(correlation_matrix, dtype=torch.float32)

    # Total number of channels
    nc = correlation_matrix.shape[0]

    # Ensure we can create exact batches
    assert nc == num_batches * batch_size, \
        f"Total channels {nc} must be exactly divisible by batch size {batch_size}"

    # Compute pairwise correlation distances
    correlation_distances = 1 - torch.abs(correlation_matrix)

    # Compute a similarity score for each channel to all other channels
    channel_similarity_scores = torch.mean(correlation_distances, dim=1)

    # Sort channels by their overall similarity (lower score means more similar to others)
    _, sorted_channel_indices = torch.sort(channel_similarity_scores)

    # Initialize batches
    batches = []

    # Distribute channels across batches
    for batch_idx in range(num_batches):
        # Select a subset of channels with good internal correlation
        batch_channels = select_batch_with_high_correlation(
            correlation_matrix,
            sorted_channel_indices,
            batch_idx,
            num_batches,
            batch_size
        )

        batches.append(batch_channels.tolist())

    return batches


def select_batch_with_high_correlation(correlation_matrix, sorted_indices, batch_idx, num_batches, batch_size):
    """
    Select a batch of channels with high internal correlation.

    Parameters:
    - correlation_matrix: Full correlation matrix
    - sorted_indices: Indices sorted by overall similarity
    - batch_idx: Current batch index
    - num_batches: Total number of batches
    - batch_size: Size of each batch

    Returns:
    - Indices of channels for the current batch
    """
    # Compute start and end indices for this batch
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size

    # Select initial batch of channels
    batch_channels = sorted_indices[start_idx:end_idx]

    # Compute internal correlation of the batch
    batch_corr = correlation_matrix[batch_channels, :][:, batch_channels]

    # If the batch doesn't have good internal correlation, try optimizing
    if batch_size > 1:
        # Compute pairwise correlations within the batch
        pairwise_corr = torch.abs(batch_corr - torch.eye(batch_size, device=batch_corr.device))
        mean_pairwise_corr = torch.mean(pairwise_corr)

        # Try to improve batch composition
        for _ in range(10):  # Limit optimization iterations
            # Find channels that might improve batch correlation
            best_swap_score = mean_pairwise_corr
            best_swap = None

            for swap_idx in range(batch_size):
                for candidate_idx in sorted_indices:
                    if candidate_idx not in batch_channels:
                        # Create a copy of current batch and swap a channel
                        temp_batch = batch_channels.clone()
                        temp_batch[swap_idx] = candidate_idx

                        # Compute new batch correlation
                        temp_batch_corr = correlation_matrix[temp_batch, :][:, temp_batch]
                        temp_pairwise_corr = torch.abs(
                            temp_batch_corr - torch.eye(batch_size, device=temp_batch_corr.device))
                        temp_mean_corr = torch.mean(temp_pairwise_corr)

                        # Update if we find a better batch
                        if temp_mean_corr > best_swap_score:
                            best_swap_score = temp_mean_corr
                            best_swap = (swap_idx, candidate_idx)

            # Apply the best swap if found
            if best_swap:
                batch_channels[best_swap[0]] = best_swap[1]
                mean_pairwise_corr = best_swap_score
            else:
                break

    return batch_channels


def create_correlation_matrix(nc, seed=42):
    """
    Create a synthetic correlation matrix using PyTorch.

    Parameters:
    - nc: Number of channels
    - seed: Random seed for reproducibility

    Returns:
    - Torch tensor correlation matrix
    """
    torch.manual_seed(seed)

    # Create a random symmetric correlation matrix
    C = torch.rand(nc, nc)
    C = (C + C.T) / 2  # Make symmetric

    # Set diagonal to 1
    torch.diagonal(C)[:] = 1.0

    return C

