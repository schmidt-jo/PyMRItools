import logging
from performance_experiments.utils import get_output_dir, create_phantom_data
from pymritools.recon.loraks_dev_cleanup.loraks import Loraks
from pymritools.recon.loraks_dev_cleanup.ac_loraks import AcLoraksOptions
from pymritools.recon.loraks_dev_cleanup.utils import (
    prepare_k_space_to_batches, unprepare_batches_to_k_space, check_channel_batch_size_and_batch_channels,
    pad_input
)


log_module = logging.getLogger(__name__)

def main():
    log_module.info(f"Create phantom")
    k = create_phantom_data((140, 128, 4, 3))

    log_module.info(f"create ac loraks object")
    ac_opts = AcLoraksOptions()

    ac_loraks = Loraks.create(ac_opts)

    log_module.info("prepare k - space")
    batch_size_channels = 2





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()