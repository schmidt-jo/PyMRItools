from pymritools.config import BaseClass
import logging
from simple_parsing import field
from dataclasses import dataclass

log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    input_file: str = field(
        alias="-i", default="",
        help="Input Nifti file."
    )
    echo_times: list = field(
        alias="-t", default_factory=lambda: [0.0],
        help="Input echo times as list."
    )
