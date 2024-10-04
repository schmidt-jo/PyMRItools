import logging
from simple_parsing import field
from pymritools.config import BaseClass
from dataclasses import dataclass

log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    input_file: str = field(
        alias="-i", default="",
        help="Input Raw data file in .dat format."
    )
    input_sequence_config: str = field(
        alias="-sc", default="",
        help="Input Pulseq sequence configuration file."
    )

