import logging
from pymritools.config import BaseClass
from dataclasses import dataclass
from simple_parsing import field
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
        alias="-ia", default="./examples/raw_data/results/k_space.pt",
        help="input affine matrix, necessary if input file is .pt to output .nii file."
    )
    in_sampling_mask: str = field(
        alias="-is", default="",
        help=f"(Optional) Input sampling mask for reconstruction masking sampled voxels in the input."
             f" If not given it will be deduced from the input."
    )
    # vars
    coil_compression: int = field(
        alias="-cc", default=32,
        help=f"Specify coil compression for multi channel data. Default working mode is "
             f"Joint-Echo-Channel reconstruction, which can lead to memory problems. "
             f"Compression can help in those cases."
    )
    read_dir: int = field(
        alias="-rd", default=0,
        help="specify read direction if not in x. Necessary for AC LORAKS to deduce AC region."
    )
    process_slice: bool = field(
        alias="-ps", default=False,
        help="toggle processing of middle slize and not whole volume, eg. for testing LORAKS parameters."
    )
    flavour: str = field(
        choices=["AC-LORAKS"], alias="-f", default="AC-LORAKS",
        help=f"LORAKS flavour. Implementation of different LORAKS variations. (not full availability yet)"
    )
    radius: int = field(
        alias="-r", default=3,
        help="Loraks neighborhood radius."
    )
    # C matrix
    c_rank: int = field(alias="-cr", default=150, help="Rank for C matrix formalism.")
    c_lambda: float = field(
        alias="-cl", default=0.1,
        help=f"regularization parameter for Loraks C matrix "
             f"rank minimization. Set 0.0 to disable C regularization."
    )
    s_rank: int = field(alias="-sr", default=250)
    s_lambda: float = field(
        alias="-sl", default=0.1,
        help=f"regularization parameter for Loraks S matrix "
             f"rank minimization. Set 0.0 to disable S regularization."
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
    use_wandb: bool = field(
        alias="-wb", default=False,
        help="Use wandb logging or optimization features."
    )
    