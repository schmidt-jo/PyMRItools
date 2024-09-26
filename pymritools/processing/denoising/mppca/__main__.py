"""
MPPCA denoising using method outlined in:
Does et al. 2019: Evaluation of principal component analysis image denoising on
multi‐exponential MRI relaxometry. Magn Reson Med
DOI: 10.1002/mrm.27658
_____
24.11.2023, Jochen Schmidt
"""
import logging
from d_mppca import d_fn, options

logging.getLogger("simple_parsing").setLevel(logging.WARNING)


def main(config: options.Config):
    d_fn.main(config=config)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("___________________________________________________________________")
    logging.info("________________________ MP PCA Denoizing  ________________________")
    logging.info("___________________________________________________________________")

    parser, prog_args = options.create_cli()

    opts = options.Config.from_cli(prog_args)
    # set logging level after possible config file read
    if opts.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        main(config=opts)

    except Exception as e:
        logging.exception(e)
        parser.print_usage()
