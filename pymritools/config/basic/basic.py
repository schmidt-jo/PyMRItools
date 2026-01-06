"""
Create base configuration class to derive from
"""
import logging
from rich.logging import RichHandler
from dataclasses import dataclass
from simple_parsing import field, ArgumentParser
from simple_parsing.helpers import Serializable
import pathlib as plib

log_module = logging.getLogger(__name__)


def setup_parser(prog_name: str, dict_config_dataclasses: dict):
    if not isinstance(dict_config_dataclasses, dict):
        err = (f"Provide a dictionary of config dataclasses (not {type(dict_config_dataclasses)}), "
               f"of structure: dest_name: dataclass.")
        log_module.error(err)
        raise AttributeError(err)

    parser = ArgumentParser(prog=prog_name)
    for name, config_dataclass in dict_config_dataclasses.items():
        parser.add_arguments(config_dataclass, dest=name)
    return parser, parser.parse_args()


def setup_program_logging(name: str, level: int = logging.INFO):
    logging.basicConfig(format='[italic sky_blue3]%(name)s[/] --  %(message)s',
                        datefmt='%I:%M:%S',
                        level=level,
                        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
                        )
    # run some exclusions we dont want to expose to the user log
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("simple_parsing").setLevel(logging.WARNING)

    num_chars_name = len(name)
    num_chars_fill = 50
    ul = "".ljust(num_chars_fill, " ")
    logging.info(f"[bold purple3 underline]{ul}[/]")
    logging.info(f"[bold purple3 underline]{name.ljust(int((num_chars_fill + num_chars_name)/ 2), ' ').rjust(num_chars_fill, ' ')}")
    logging.info(f"[bold purple3 underline]{ul}[/]")


@dataclass
class BaseClass(Serializable):
    config_file: str = field(
        alias="-c", default="",
        help="Input configuration file (.json) covering entries to this Settings object."
    )
    out_path: str = field(
        alias="-o", default="",
        help="Path to output directory."
    )
    # flags
    use_gpu: bool = field(
        alias="-gpu", default=True,
        help="Set device (gpu or cpu) for accelerated computations."
    )
    gpu_device: int = field(
        alias="-gpud", default=0,
        help="Set gpu device if multiple are available."
    )
    visualize: bool = field(
        alias="-v", default=True,
        help="Toggle visualizations"
    )
    debug: bool = field(
        alias="-d", default=False,
        help="Toggle debugging mode, and logging debug level."
    )
    slurm: bool = field(
        default=False,
        help="Toggle slurm mode, disables visualizations and progress bars and activates some additional logging"
    )

    def _check_non_default_vars(self) -> (dict, dict):
        # create default class instance
        def_config = self.__class__()
        # store non default arguments inside dictionary
        non_default_config = {}
        for key, value in vars(self).items():
            if self.__getattribute__(key) != def_config.__getattribute__(key):
                non_default_config.__setitem__(key, value)
        return non_default_config

    @classmethod
    def from_cli(cls, args: ArgumentParser.parse_args, parser: ArgumentParser = None):
        """
        Create settings from command line interface arguments via simple parsing.
        Check for configuration file input but prioritizes additional cli input
        if given explicitly and varying from default.
        :param args: simple-parsing parsed arguments
        :param parser: simple-parsing argument parser - (optional) will provide help message in case of error
        :return:
        """
        # create class instance via the args
        instance = args

        # here we find explicit cmd line args (if not defaults)
        non_default_config = instance._check_non_default_vars()

        # check for config file
        if instance.config_file:
            file = plib.Path(instance.config_file).absolute()
            if file.is_file():
                log_module.debug(f"load config file {file}")
                # classmethod, will create new instance
                instance = instance.load(file)
                # overwrite non default input args (assumed to be explicit, will not catch explicitly given default args)
                for key, value in non_default_config.items():
                    instance.__setattr__(key, value)
        if not instance.out_path:
            err = "No Output Path provided!"
            log_module.error(err)
            if parser is not None:
                parser.print_help()
            raise ValueError(err)
        return instance

    def display(self):
        # display via logging
        s = "[bold yellow underline]          Config         [/]\n"
        for k, v in self.to_dict().items():
            s += f"\t\t\t[yellow]{k}[/]:".ljust(30) + f"{v}\n".rjust(55, ".")
        log_module.info(s)
