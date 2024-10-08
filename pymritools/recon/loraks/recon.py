import logging
from pymritools.config.recon import PyLoraksConfig
from pymritools.config import setup_program_logging, setup_parser

log_module = logging.getLogger(__name__)


def recon(settings: PyLoraksConfig):
    pass


def main():
    # setup  logging
    setup_program_logging(name="PyLoraks", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(prog_name="PyLoraks", dict_config_dataclasses={"settings": PyLoraksConfig})
    # get cli args
    settings = PyLoraksConfig.from_cli(args=args.settings, parser=parser)
    settings.display()

    try:
        recon(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()
