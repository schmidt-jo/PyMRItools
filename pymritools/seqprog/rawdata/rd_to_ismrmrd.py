import logging
from pymritools.config.seqprog import RD, Sampling, PulseqParameters2D
from pymritools.config import setup_program_logging, setup_parser
from pymritools.seqprog.rawdata.load_fns import load_pulseq_rd
import pathlib as plib
import twixtools
log_module = logging.getLogger(__name__)


def rd_to_ismrmrd(config: RD):
    # load sampling configuration
    path_to_file = plib.Path(config.input_sample_config).absolute()
    if not path_to_file.exists():
        err = f"File {path_to_file.as_posix()} does not exist."
        log_module.error(err)
        raise FileNotFoundError(err)
    sampling = Sampling.load(path_to_file.as_posix())

    # load sequence config
    path_to_file = plib.Path(config.input_sequence_config).absolute()
    if not path_to_file.exists():
        err = f"File {path_to_file.as_posix()} does not exist."
        log_module.error(err)
        raise FileNotFoundError(err)
    pulseq = PulseqParameters2D.load(path_to_file.as_posix())

    # load data
    path_to_file = plib.Path(config.input_file).absolute()
    if not path_to_file.exists():
        err = f"File {path_to_file.as_posix()} does not exist."
        log_module.error(err)
        raise FileNotFoundError(err)
    log_module.info(f"Loading raw data file: {path_to_file.as_posix()}")
    if not ".dat" in path_to_file.suffixes:
        err = "File not recognized as .dat raw data file."
        log_module.error(err)
        raise ValueError(err)
    twix = twixtools.read_twix(path_to_file.as_posix(), parse_geometry=True, verbose=True, include_scans=-1)[0]
    geometry = twix["geometry"]
    data_mdbs = twix["mdb"]
    hdr = twix["hdr"]
    log_module.info("Loading RD")
    load_pulseq_rd(
        pulseq_config=pulseq, sampling_config=sampling,
        data_mdbs=data_mdbs, geometry=geometry, hdr=hdr
    )
    log_module.info("Done")


def main():
    # setup logging
    setup_program_logging(name="Raw Data to ISMRMRD", level=logging.INFO)

    # setup parser
    parser, args = setup_parser(prog_name="Raw Data to ISMRMRD", dict_config_dataclasses={"settings": RD})

    # get config
    rd_config = RD.from_cli(args=args.settings, parser=parser)
    rd_config.display()

    try:
        rd_to_ismrmrd(config=rd_config)
    except Exception as e:
        parser.print_help()
        logging.exception(e)


if __name__ == '__main__':
    main()