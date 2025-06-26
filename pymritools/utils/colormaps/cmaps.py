import logging
import pathlib as plib
import plotly.colors as plc
import json

logger = logging.getLogger(__name__)


def _available():
    source_path = plib.Path(__file__).absolute().parent
    available_colorscales = [f.stem for f in source_path.iterdir() if f.is_file() and f.name.endswith(".json")]
    return available_colorscales


def check_colormap_available(name: str):
    source_path = plib.Path(__file__).absolute().parent
    available_colorscales = _available()
    if name in available_colorscales:
        return True
    return False


def show_available_colormaps():
    acm = _available()
    logger.info(f"Available colormaps: {acm}")


def get_colormap(name: str):
    source_path = plib.Path(__file__).absolute().parent

    if not check_colormap_available(name):
        msg = f"colorscale {name} not available."
        show_available_colormaps()
        logger.error(msg)
        raise ValueError(msg)

    # load rgb data list
    f = source_path.joinpath(name).with_suffix(".json")
    with open(f, "r") as j_file:
        cm_data = json.load(j_file)

    cm_data = [plc.label_rgb(c) for c in cm_data]
    return plc.make_colorscale(cm_data)


if __name__ == '__main__':
    cmap_navia = get_colormap("navia")
    print(cmap_navia)
