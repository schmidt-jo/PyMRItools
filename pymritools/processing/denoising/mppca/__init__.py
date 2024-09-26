import pathlib as plib
import sys

autodmri_path = plib.Path(__file__).absolute().parent.parent.joinpath("autodmri/")
sys.path.append(autodmri_path.as_posix())

from .options import Config
from .d_fn import denoise
