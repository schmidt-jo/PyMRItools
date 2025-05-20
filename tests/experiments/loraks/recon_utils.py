import logging

import torch

from tests.experiments.utils import create_phantom_data, get_output_dir
from pymritools.recon.loraks_dev_cleanup.utils import (
    prepare_k_space_to_batches, unprepare_batches_to_k_space, check_channel_batch_size_and_batch_channels,
    pad_input, unpad_output
)

log_module = logging.getLogger(__name__)


def prepare_and_unprepare_k_space():
    out_dir = get_output_dir("recon_utils")
    # create phantom and add slice dimension
    phantom = create_phantom_data(shape_xyct=(140, 128, 4, 3)).unsqueeze(2)
    # chose a channel batch size
    batch_size_channels = 2
    # do the channel batching
    batch_channel_indices = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=phantom, batch_size_channels=batch_size_channels
    )

    # do the batch preparation
    k_batched, input_shape = prepare_k_space_to_batches(
        k_space_rpsct=phantom, batch_channel_indices=batch_channel_indices
    )
    log_module.info(f"prepared batched data shape: {k_batched.shape}")
    # pad the input
    k_batched, padding = pad_input(k_batched)

    # do the reverse
    k_batched = unpad_output(k_batched, padding=padding)
    k_output = unprepare_batches_to_k_space(
        k_batched=k_batched, batch_channel_indices=batch_channel_indices, original_shape=input_shape
    )
    log_module.info(f"output data shape: {k_output.shape}")
    assert torch.allclose(k_output, phantom)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    prepare_and_unprepare_k_space()
