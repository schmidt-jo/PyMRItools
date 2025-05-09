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
    radius: int = field(
        alias="-r", default=3,
        help="Loraks neighborhood radius."
    )
    rank: int = field(alias="-rank", default=250)
    reg_lambda: float = field(
        alias="-l", default=0.0,
        help=f"regularization parameter for Loraks matrix "
             f"rank minimization. Set 0.0 to use true data consistency (default)."
    )
    matrix_type: str = choice(
        "S", "C", alias="-t", default="S",
        help="LORAKS matrix type to be used"
    )
    # lambda_data: float = field(alias="-dl", default=0.5)
    conv_tol: float = field(
        alias="-tol", default=1e-3,
        help="Convergence tolerance for the conjugate gradient algorithm."
    )
    max_num_iter: int = field(
        alias="-mni", default=10,
        help="Maximum number of iterations if not converged."
    )

    batch_size: int = field(
        alias="-b", default=4,
        help="Batch size for gpu computation. Reduce if out of memory."
    )
    