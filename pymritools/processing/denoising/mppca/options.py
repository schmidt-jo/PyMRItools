import dataclasses as dc
import simple_parsing as sp
import pathlib as plib
import logging
import pandas as pd

log_module = logging.getLogger(__name__)


@dc.dataclass
class Config(sp.helpers.Serializable):
    """
    Configuration for denoising
    """
    config_file: str = sp.field(
        alias="-c", default="../examples/config.json",
        help="provide Configuration file (.json)"
    )
    in_path: str = sp.field(
        alias="-i", default="",
        help="set filepath to .nii or .pt file"
    )
    in_affine: str = sp.field(
        alias="-ia", default="",
        help="input affine matrix, necessary if input file is .pt, optional if .nii"
    )
    save_path: str = sp.field(
        alias="-s", default="",
        help="set path to save files (optional, if blank use input path)"
    )
    file_prefix: str = sp.field(
        default="d", alias="-fp",
        help=f"Output file prefix appended to name after denoising / debiasing"
    )
    use_gpu: bool = sp.field(
        default=True, alias="-gpu", help="try using gpu processing"
    )
    gpu_device: int = sp.field(
        default=0, alias="-gpud", help="specify which gpu to use if applicable, omitted if use_gpu=False"
    )
    debug: bool = sp.field(
        default=False, alias="-d", help="toggle logging debug information"
    )
    fixed_p: int = sp.field(
        default=0, alias="-p", help="(optional) fix the number of singular values to keep in patch."
                                    "For (default) 0 the number is computed per patch from the MP inequality."
    )
    normalize: bool = sp.field(
        default=False, alias="-n", help="(optional), normalize data (across t dimension) to max 1 before pca"
    )
    input_image_data: bool = sp.field(
        default=False, alias="-iimg", help="if input is in image space set to true. "
                                           "Otherwise input is assumed to be k-space data"
    )
    noise_bias_correction: bool = sp.field(
        default=False, alias="-nbc",
        help="(optional) noise bias correction "
             "using stationary or non stationary noise estimates and "
             "assuming non-central chi noise distribution."
    )
    noise_bias_mask: str = sp.field(
        default="", alias="-nbm", help="input noise mask for noise statistics estimation if bias correction is set."
    )

    @classmethod
    def from_cli(cls, args: sp.ArgumentParser.parse_args):
        instance = args.config
        if instance.config_file:
            c_path = plib.Path(instance.config_file).absolute()
            if not c_path.is_file():
                err = f"Config File set: {c_path.as_posix()} not found!"
                log_module.error(err)
                raise FileNotFoundError(err)
            instance = cls.load(c_path.as_posix())
        if not instance.save_path:
            instance.save_path = plib.Path(instance.in_path).absolute().parent.as_posix()
        instance.display()
        return instance

    def display(self):
        # display via logging
        df = pd.Series(self.to_dict())
        # concat empty entry to start of series for nicer visualization
        df = pd.concat([pd.Series([""], index=["___ Config ___"]), df])
        # display
        log_module.info(df)


def create_cli() -> (sp.ArgumentParser, sp.ArgumentParser.parse_args):
    """
        Build the parser for arguments
        Parse the input arguments.
        """
    parser = sp.ArgumentParser(prog='denoising_mppca')
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    return parser, args
