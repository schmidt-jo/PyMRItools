from pymritools.simulation.emc import sequence
from pymritools.config.emc import EmcParameters, EmcSimSettings
from pymritools.config.database import DB
from pymritools.utils import setup_program_logging
import logging
import pathlib as plib
import simple_parsing
log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


def core_sim(params: EmcParameters, settings: EmcSimSettings) -> DB:
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
    db = DB.from_simulation_data(params=params, sim_data=sim_obj.data)
    # plot stuff
    if settings.visualize:
        # plot magnetization profile snapshots
        # sim_obj.plot_magnetization_profiles(animate=False)
        # sim_obj.plot_emc_signal()
        # if settings.signal_fourier_sampling:
        #     sim_obj.plot_signal_traces()
        # plot database
        db.plot(sim_obj.fig_path)
    return db


def cli_sim(params: EmcParameters, settings: EmcSimSettings):
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
    setup_program_logging(name="EMC Simulation", level=logging.INFO)

    parser = simple_parsing.ArgumentParser(prog='emc_sim')
    parser.add_arguments(EmcParameters, dest="params")
    parser.add_arguments(EmcSimSettings, dest="settings")
    prog_args = parser.parse_args()

    settings = EmcSimSettings.from_cli(prog_args.settings)
    params = prog_args.params

    if prog_args.settings.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cli_sim(settings=settings, params=params)

    except Exception as e:
        logging.exception(e)
        parser.print_usage()


if __name__ == '__main__':
    main()
