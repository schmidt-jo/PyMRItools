"""
Create base configuration class to derive from
"""
import logging
from dataclasses import dataclass
from simple_parsing import field, ArgumentParser
from simple_parsing.helpers import Serializable
import pathlib as plib

log_module = logging.getLogger(__name__)


@dataclass
class BaseClass(Serializable):
    config_file: str = field(
        alias="-c", default="",
        help="Input configuration file (.json) covering entries to this Settings object."
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
    def from_cli(cls, args: ArgumentParser.parse_args):
        """
        Create settings from command line interface arguments via simple parsing.
        Check for configuration file input but prioritizes additional cli input
        if given explicitly and varying from default.
        :param args: simple-parsing parsed arguments dictionary
        :return:
        """
        # create class instance via the args
        instance = cls.from_dict(args.to_dict())

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
        return instance

    def display(self):
        # display via logging
        s = "___ Config ___\n"
        for k, v in self.to_dict().items():
            s += f"\t\t\t{k}: {v}\n"
        log_module.info(s)