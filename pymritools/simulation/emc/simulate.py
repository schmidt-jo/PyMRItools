import sequence
from pymritools.config.emc import EmcParameters, EmcSettings
from pymritools.config.database import DB
import logging
import pathlib as plib
import simple_parsing
log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


# def sim(
#         emc_seq_file: str = "", pypsi_path: str = "", pulse_file: str = "",
#         sim_type: str = "megesse",
#         resample_pulse_to_dt_us: float = 5.0,
#         use_gpu: bool = False, gpu_device: int = 0,
#         visualize: bool = False, debug: bool = False,
#         sample_number: int = 1000, length_z: float = 0.005,
#         t2_list: list = None, b1_list: list = None) -> db.DB:
#     """
#     Function to be called when using simulation aspect of package from another python application.
#     I.E. optimization or ai model training.
#     basically just create the parameter options from kwargs and passing it onto core
#     """
#     config = EmcSettings(
#         emc_params_file=emc_seq_file, pulse_file=pulse_file,
#         save_path="", database_name="_", sim_type=sim_type, signal_fourier_sampling=False,
#         visualize=visualize, debug=debug, resample_pulse_to_dt_us=resample_pulse_to_dt_us,
#         use_gpu=use_gpu, gpu_device=gpu_device
#     )
#     settings = SimulationData(
#         sample_number=sample_number, length_z=length_z,
#         t1_list=[1.5], t2_list=t2_list, b1_list=b1_list
#     )
#     sim_params = options.SimulationParameters(config=config, settings=settings)
#     db = core_sim(sim_params=sim_params)
#     return db


def core_sim(params: EmcParameters, settings: EmcSettings) -> DB:
    """
    core simulation and plotting
    """
    settings.display()

    if settings.sim_type.startswith("mese"):
        sim_obj = sequence.MESE(params=params, settings=settings)
    elif settings.sim_type == "megesse":
        sim_obj = sequence.MEGESSE(params=params, settings=settings)
    # elif settings.sim_type == "megessevesp":
    #     sim_obj = sequence.MEGESSEVESP(params=params)
    # if sim_settings.sim_type == "single":
    #     sim_obj = simulations.(sim_params=sim_params, device=device)
    else:
        err = f"sequence type choice ({settings.sim_type}) not implemented for simulation"
        log_module.error(err)
        raise ValueError(err)
    # simulate sequence
    sim_obj.simulate()
    # create database
    db = DB.build_from_sim_data(sim_params=params, sim_data=sim_obj.data)
    # plot stuff
    if settings.visualize:
        # plot magnetization profile snapshots
        sim_obj.plot_magnetization_profiles(animate=False)
        sim_obj.plot_emc_signal()
        if settings.signal_fourier_sampling:
            sim_obj.plot_signal_traces()
        # plot database
        db.plot(sim_obj.fig_path)
    return db


def cli_sim(params: EmcParameters, settings: EmcSettings):
    """
    Function to be called when using tool as cmd line interface, passing the cli created options.
    Just doing the core sim and saving the data
    """
    db = core_sim(params=params, settings=settings)
    # save files
    save_path = plib.Path(settings.save_path).absolute()
    if settings.config_file:
        c_name = plib.Path(settings.config_file).absolute().stem
    else:
        c_name = "emc_config"
    save_file = save_path.joinpath(c_name).with_suffix(".json")
    logging.info(f"Save Config File: {save_file.as_posix()}")
    settings.save_json(save_file.as_posix(), indent=2)
    # database
    save_file = save_path.joinpath(settings.database_name)
    logging.info(f"Save DB File: {save_file.as_posix()}")
    db.save(save_file)


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("__________________________________________________________")
    logging.info("__________________ EMC torch simulation __________________")
    logging.info("__________________________________________________________")

    parser = simple_parsing.ArgumentParser(prog='emc_sim')
    parser.add_arguments(EmcParameters, dest="params")
    parser.add_arguments(EmcSettings, dest="settings")
    prog_args = parser.parse_args()

    # opts = options.SimulationParameters.from_cli(prog_args)
    # set logging level after possible config file read
    if prog_args.settings.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cli_sim(settings=prog_args.settings, params=prog_args.params)

    except Exception as e:
        logging.exception(e)
        parser.print_usage()


if __name__ == '__main__':
    main()
