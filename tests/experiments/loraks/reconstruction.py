import logging

from tests.experiments.utils import get_output_dir
from pymritools.config.recon.loraks import Settings
from pymritools.recon.loraks_dev_cleanup.recon import main as reco


def main():
    path_out = get_output_dir("loraks_recon")
    config = Settings(
        loraks_algorithm="AC-LORAKS",
        out_path=path_out.as_posix(),
        in_k_space="C:\\Daten_Jo\\Daten\\03_work\\06_owncloud\\data\\data\\dev\\loraks\\phantom_k_us_cs.pt",
        max_num_iter=500
    )
    reco(config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
