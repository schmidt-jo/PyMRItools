import logging
import pathlib as plib
import sys
import torch

from simple_parsing import ArgumentParser
from dataclasses import dataclass

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.experiments.loraks.comparison_memory import torch_loraks_run
from tests.utils import get_test_result_output_dir, ResultMode

logger = logging.getLogger(__name__)


@dataclass
class TorchCPU:
    file: str = None
    rank: int = -1
    regularization_lambda: float = -1.0
    max_num_iter: int = -1


def set_parser() -> (ArgumentParser, TorchCPU):
    parser = ArgumentParser(prog="Torch Memory Profile CPU")
    parser.add_arguments(TorchCPU, dest="config")
    return parser, parser.parse_args().config


def recon_ac_loraks_cpu(
        k: torch.Tensor,
        rank: int, regularization_lambda: float,
        max_num_iter: int = 30,):
    device = torch.device("cpu")
    logger.info(f"Set device: {device}")

    _ = torch_loraks_run(
        k=k, device=device, rank=rank, regularization_lambda=regularization_lambda, max_num_iter=max_num_iter
    )


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    parser, config = set_parser()
    fn = plib.Path(config.file).absolute()

    if not fn.exists():
        raise FileNotFoundError(fn)

    # k = torch.randn((1000, 1000))
    k = torch.load(fn)

    recon_ac_loraks_cpu(
        k=k, rank=config.rank,
        regularization_lambda=config.regularization_lambda, max_num_iter=config.max_num_iter
    )
