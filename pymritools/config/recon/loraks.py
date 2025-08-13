import logging
from pymritools.config import BaseClass
from dataclasses import dataclass
from simple_parsing import field, choice
log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    """
    Configuration for PyLORAKS reconstruction
    """
    in_k_space: str = field(
        alias="-ik", default="./examples/raw_data/results/k_space.pt",
        help="set filepath to .pt file"
    )
    in_affine: str = field(
        alias="-ia", default="",
        help="input affine matrix, necessary if input file is .pt to output .nii file."
    )
    process_slice: bool = field(
        alias="-ps", default=False,
        help="toggle processing of middle slize and not whole volume, eg. for testing LORAKS parameters."
    )
    patch_size: int = field(
        alias="-p", default=5,
        help="Loraks neighborhood square patch size."
    )
    rank: int = field(alias="-rank", default=180)
    reg_lambda: float = field(
        alias="-l", default=0.0,
        help=f"regularization parameter for Loraks matrix "
             f"rank minimization. Set 0.0 to use true data consistency (default)."
    )
    loraks_algorithm: str = choice(
        "AC-LORAKS", "P-LORAKS", "AC-LORAKS-ls", alias="-alg", default="AC-LORAKS-ls",
        help="LORAKS algorithm to be used"
    )
    matrix_type: str = choice(
        "S", "C", alias="-t", default="S",
        help="LORAKS matrix type to be used"
    )
    conv_tol: float = field(
        alias="-tol", default=1e-3,
        help="Convergence tolerance for the conjugate gradient algorithm."
    )
    max_num_iter: int = field(
        alias="-mni", default=200,
        help="Maximum number of iterations if not converged."
    )
    batch_size: int = field(
        alias="-b", default=-1,
        help="Batch size for gpu computation. Reduce if out of memory."
    )

    