import torch


def batch_channels_by_correlation(correlation_matrix, num_batches, batch_size):
    """
    Create batches of channels based on their correlation similarities.

    Parameters:
    - correlation_matrix: NumPy array of shape [nc, nc]
    - num_batches: Number of batches to create
    - batch_size: Number of channels per batch

    Returns:
    - List of batches, where each batch is a list of channel indices
    """
    # Take the absolute correlation values to consider both positive and negative correlations
    abs_corr = torch.abs(correlation_matrix)

    # Convert correlation matrix to a distance matrix
    # Lower correlation means higher distance
    distance_matrix = 1 - abs_corr

    # Use K-means clustering to group similar channels
    kmeans = KMeans(n_clusters=num_batches, random_state=42)

    # Perform clustering based on the distance matrix
    # We'll use the distance matrix as a proxy for channel similarity
    kmeans.fit(distance_matrix)

    # Get cluster labels for each channel
    cluster_labels = kmeans.labels_

    # Create batches based on cluster assignments
    batches = []
    for batch_idx in range(num_batches):
        # Find channels belonging to this batch
        batch_channels = torch.where(cluster_labels == batch_idx)[0]

        # Ensure we have exactly batch_size channels
        if len(batch_channels) > batch_size:
            # If too many, select the batch_size most correlated channels within this cluster
            batch_channels = select_most_correlated_subset(correlation_matrix, batch_channels, batch_size)
        elif len(batch_channels) < batch_size:
            # If too few, pad with most similar additional channels
            batch_channels = pad_batch(correlation_matrix, batch_channels, batch_size)

        batches.append(batch_channels.tolist())

    return batches



def select_most_correlated_subset(correlation_matrix, candidate_channels, target_size):
    """
    Select a subset of channels with the highest internal correlations.
    """
    # Create a submatrix of correlations for candidate channels
    sub_corr = correlation_matrix[torch.ix_(candidate_channels, candidate_channels)]

    # Compute mean correlation for each channel with other channels in the subset
    mean_correlations = torch.mean(torch.abs(sub_corr), axis=1)

    # Select channels with highest mean correlations
    top_indices = torch.argsort(mean_correlations)[-target_size:]

    return candidate_channels[top_indices]


def pad_batch(correlation_matrix, current_batch, target_size):
    """
    Pad the batch with most similar channels from outside the current batch.
    """
    # Find all channels not in the current batch
    all_channels = set(range(correlation_matrix.shape[0]))
    current_batch_set = set(current_batch)
    remaining_channels = list(all_channels - current_batch_set)

    # Compute mean correlation of remaining channels to current batch
    batch_mean_correlation = torch.mean(
        torch.abs(correlation_matrix[torch.ix_(current_batch, remaining_channels)]),
        axis=0
    )

    # Select most correlated channels to pad the batch
    additional_channels = torch.argsort(batch_mean_correlation)[-(target_size - len(current_batch):]

    return torch.concatenate([current_batch, [remaining_channels[i] for i in additional_channels]])

